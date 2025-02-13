import numpy as np
import torch
import cv2
from PIL import Image
from torchvision import transforms as T
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detr.image_processing import preprocess_image, plot_detr_results
from detr.model_detr import COCO_INSTANCE_CATEGORY_NAMES, load_detr_model

def generate_kitti_txt(image_path, model, output_txt_path):
    image, image_tensor = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(image_tensor)
    
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.5  # Umbral de confianza
    bboxes = outputs['pred_boxes'][0, keep].numpy()
    labels = probas[keep].argmax(-1).cpu().numpy()
    scores = probas.max(-1).values[keep].cpu().numpy()

    height, width = image.size

    with open(output_txt_path, 'w') as f:
        for bbox, label, score in zip(bboxes, labels, scores):
            if label >= len(COCO_INSTANCE_CATEGORY_NAMES):
                print(f"⚠️ Advertencia: Se encontró un label fuera de rango ({label}), será ignorado.")
                continue  # Ignorar esta detección
            
            cx, cy, w, h = bbox
            x0, y0 = int((cx - w / 2) * width), int((cy - h / 2) * height)
            x1, y1 = int((cx + w / 2) * width), int((cy + h / 2) * height)

            line = f"{COCO_INSTANCE_CATEGORY_NAMES[label]} 0.00 0 -10 {x0} {y0} {x1} {y1} -1 -1 -1 -1 -1 -1 -1 {score:.4f}\n"
            f.write(line)
    
    plot_detr_results(image, bboxes, labels)  # Mostrar la imagen con los resultados

# Cargar el modelo DETR
detr_model = load_detr_model()

# Ruta de la imagen y del archivo de salida
image_path = "../images/obj_det_images/000002.png"
output_txt_path = "../labels/pred/000002.txt"

generate_kitti_txt(image_path, detr_model, output_txt_path)
