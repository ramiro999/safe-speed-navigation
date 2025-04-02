# inference.py
import argparse
import multiprocessing as mp
import os
import torch
import cv2
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image

from nmrf.config import get_cfg
from nmrf.utils.logger import setup_logger
from nmrf.data import datasets
from nmrf.utils import frame_utils
from nmrf.utils import visualization
from nmrf.models import build_model
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)visualized_output
    if args.config_file and len(args.config_file) > 0:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="NMRF demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--dataset-name", help="Dataset name to generate prediction results")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input image pairs; "
             "or a pair of single glob pattern such as 'directory/left/*.jpg directory/right/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save prediction results.",
    )
    parser.add_argument(
        "--show-attr",
        default="disparity",
        help="The attribute to visualize.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

@torch.no_grad()
def run_on_dataset(dataset, model, output, find_output_path=None, show_attr="disparity"):
    model.eval()
    disp_preds = []

    for idx in range(len(dataset)):
        sample = dataset[idx]
        rgb = sample["img1"].permute(1, 2, 0).numpy()
        viz = visualization.Visualizer(rgb)

        sample["img1"] = sample["img1"][None]
        sample["img2"] = sample["img2"][None]
        result_dict = model(sample)

        if show_attr == "error":
            valid = sample["valid"]
            disp_gt = sample["disp"]
            disp_pred = result_dict["disp"][0].to(disp_gt.device)
            error = torch.abs(disp_pred - disp_gt).abs()
            valid = valid & (disp_gt > 0) & (disp_gt < cfg.TEST.EVAL_MAX_DISP[0])
            error[~valid] = 0
            visualized_output = viz.draw_error_map(error)

        elif show_attr == "disparity":
            disp_pred = result_dict["disp"][0].cpu()
            visualized_output = viz.draw_disparity(disp_pred, colormap="kitti")
            disp_image = visualized_output.get_image()

        else:
            raise ValueError(f"not supported visualization attribute {show_attr}")

        file_path = dataset.image_list[idx][0]

        if output:
            assert find_output_path is not None
            output_path = os.path.join(output, find_output_path(file_path))
            dirname = os.path.dirname(output_path)
            os.makedirs(dirname, exist_ok=True)

            # Guardar .npy y visualización
            np.save(output_path.replace('.png', '.npy'), disp_pred)
            visualized_output.save(output_path)

            # Si estamos mostrando disparidad, agregamos la colorbar
            if show_attr == "disparity":
                def gen_kitti_cmap():
                    map = np.array([[0, 0, 0, 114],
                                    [0, 0, 1, 185],
                                    [1, 0, 0, 114],
                                    [1, 0, 1, 174],
                                    [0, 1, 0, 114],
                                    [0, 1, 1, 185],
                                    [1, 1, 0, 114],
                                    [1, 1, 1, 0]])
                    bins = map[:-1, 3]
                    cbins = np.cumsum(bins)
                    cbins = cbins[:-1] / cbins[-1]
                    nodes = np.concatenate([np.array([0]), cbins, np.array([1])])
                    colors = map[:, :3] / 1.0
                    return mpl.colors.LinearSegmentedColormap.from_list("kitti", list(zip(nodes, colors)))

                kitti_cmap = gen_kitti_cmap()

                # Crear colorbar
                fig, ax = plt.subplots(figsize=(6, 0.6))
                fig.subplots_adjust(left=0.03, right=0.97, top=0.8, bottom=0.3)
                cb = mpl.colorbar.ColorbarBase(ax, cmap=kitti_cmap, orientation='horizontal', norm=mpl.colors.Normalize(vmin=disp_pred.min(), vmax=disp_pred.max()))
                cb.set_label('Disparity', fontsize=10, labelpad=6)
                cb.ax.tick_params(labelsize=8, length=0)
                colorbar_path = output_path.replace('.png', '_colorbar.png')
                fig.savefig(colorbar_path, bbox_inches='tight', pad_inches=0, dpi=300, transparent=True)
                plt.close(fig)

                # Unir imagen + colorbar
                disp_img_pil = Image.fromarray(disp_image)
                colorbar_img = Image.open(colorbar_path).convert("RGB").resize((disp_img_pil.width, 50))
                combined = np.vstack((np.array(disp_img_pil), np.array(colorbar_img)))
                final_with_bar_path = output_path.replace('.png', '_with_colorbar.png')
                Image.fromarray(combined).save(final_with_bar_path)

        else:
            cv2.namedWindow(f"{show_attr}", cv2.WINDOW_NORMAL)
            cv2.imshow(f"{show_attr}", visualized_output.get_image()[:, :, ::-1])
            if cv2.waitKey(0) == 27:
                break

        disp_preds.append(disp_pred)

    return disp_preds



@torch.no_grad()
def create_kitti_submission(model, image_set, output):
    training_mode = model.training
    model.eval()
    test_dataset = datasets.KITTI(split='testing', image_set=image_set)

    output_path = os.path.join(output, f'{image_set}_submission')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        sample = test_dataset[test_id]
        frame_id = sample['meta']
        sample = {"img1": sample['img1'][None], "img2": sample['img2'][None]}

        results_dict = model(sample)

        disp = results_dict['disp'][0].cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)

        frame_utils.writeDispKITTI(output_filename, disp)

    model.train(training_mode)


def _find_output_path(root):
    def wrapper(file_path):
        index = file_path.find(root)
        file_path = file_path[index:].replace(f"{root}/", "")
        return file_path
    return wrapper

# inference.py 
def run_inference(dataset_name, output, resume_path, image_list, show_attr="disparity"):
    # Crear un objeto Namespace con los parámetros necesarios para setup_cfg
    args = argparse.Namespace(
        config_file="",  # Aquí puedes especificar el archivo de configuración si lo tienes
        opts=[],
        dataset_name=dataset_name,
        output=output,
        show_attr=show_attr
    )

    # Crear cfg usando setup_cfg
    cfg = setup_cfg(args)

    # Construir el modelo
    model = build_model(cfg)[0]
    model = model.to(torch.device("cpu"))
    checkpoint = torch.load(resume_path, map_location="cpu")
    weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(weights, strict=cfg.SOLVER.STRICT_RESUME)

    # Inferencia estéreo usando la lista de imagénes proporcionadas
    dataset = datasets.KITTI(split="testing", image_set='KITTI_2015')
    dataset.image_list = image_list
    dataset.is_test = True
    dataset.extra_info = [None] * len(image_list)

    # Ejecutar la inferencia en el dataset creado con las imágenes del usuario
    disp_preds = run_on_dataset(dataset, model, output, _find_output_path(os.path.dirname(image_list[0][0])), show_attr)

    
    return disp_preds