import numpy as np
import torch
import cv2
from PIL import Image
import os
import sys
import glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rt_detr.image_processing import plot_detr_results
from rt_detr.model_rt_detr import load_rtdetr_model

# Mapeo específico para etiquetas KITTI
KITTI_CATEGORY_NAME_MAPPING = {
    "person": "Pedestrian",
    "car": "Car",
    "bicycle": "Cyclist",
    "truck": "Truck"
}

def generate_kitti_txt(image_path, model, image_processor, output_txt_path, conf_threshold=0.8):
    image_pil = Image.open(image_path).convert("RGB")
    inputs = image_processor(images=image_pil, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([[image_pil.height, image_pil.width]])
    results = image_processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=conf_threshold
    )[0]

    bboxes = results["boxes"].cpu().numpy()
    labels = results["labels"].cpu().numpy()
    scores = results["scores"].cpu().numpy()

    with open(output_txt_path, 'w') as f:
        for bbox, label, score in zip(bboxes, labels, scores):
            category_name = model.config.id2label[label]
            category_name_mapped = KITTI_CATEGORY_NAME_MAPPING.get(category_name, category_name)

            x0, y0, x1, y1 = map(int, bbox)
            line = f"{category_name_mapped} 0.00 0 0 {x0} {y0} {x1} {y1} 0 0 0 0 0 0 0 {score:.4f}\n"
            with open(output_txt_path, 'a') as f:
                f.write(line)

    plot_detr_results(image_pil, bboxes, labels, model)

# Cargar modelo e image_processor
rt_detr_model, image_processor = load_rtdetr_model()

image_folder = "../images/obj_det_images/"
output_folder = "../labels/pred/finetuned-detr/"

image_paths = glob.glob(os.path.join(image_folder, "*.png"))

for image_path in image_paths:
    image_name = os.path.basename(image_path)
    output_txt_path = os.path.join(output_folder, image_name.replace(".png", ".txt"))
    generate_kitti_txt(image_path, rt_detr_model, image_processor, output_txt_path)
