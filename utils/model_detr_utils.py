import numpy as np
import torch
import cv2
from PIL import Image
from torchvision import transforms as T
import os
import sys
from shapely.geometry import box
import glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detr.image_processing import preprocess_image, plot_detr_results
from detr.model_detr import COCO_INSTANCE_CATEGORY_NAMES, load_detr_model


"""
Este script obtiene el .txt con la información que permite medir las metricas de deteccion de objetos (IoU y mAP)
mediante el primer modelo DETR siendo de importancia el Tipo del objeto y las coordenadas de las bb en la imagen
"""

# Map COCO category names to desired names
CATEGORY_NAME_MAPPING = {
    "person": "Pedestrian",
    "bicycle": "Cyclist",
    "car": "Car",
    "truck": "Truck"
}

# Update COCO_INSTANCE_CATEGORY_NAMES with the new names
COCO_INSTANCE_CATEGORY_NAMES = [
    CATEGORY_NAME_MAPPING.get(name, name) for name in COCO_INSTANCE_CATEGORY_NAMES
]

# Calcular Intersection over Union (IoU)
def compute_iou(box1, box2):
    b1 = box(*box1)  # Crear un objeto de tipo box con las coordenadas
    b2 = box(*box2)
    intersection = b1.intersection(b2).area
    union = b1.union(b2).area
    return intersection / union if union > 0 else 0

def read_dontcare_regions(label_file_path):
    '''
    Lee las regiones 'Don't Care' desde un archivo de etiquetas KITTI.

    Args:
        label_file_path(str): Ruta del archivo .txt de etiquetas KITTI.

    Returns:
        list: Lista de bounding boxes (x0, y0, x1, y1) de las regiones "Don't Care".
    '''
    dontcare_boxes = []
    with open(label_file_path, 'r') as f:
        for line in f.readlines():
            values = line.split()
            if values[0] == "DontCare": # Identidicar regiones "Don't Care"
                x0, y0, x1, y1 = map(int, map(float, values[4:8])) # Convertir a enteros
                dontcare_boxes.append((x0, y0, x1, y1))
    return dontcare_boxes

def is_in_dontcare(x0, y0, x1, y1, dontcare_boxes):
    """
    Verifica si una bbox esta dentro de una region DontCare usando IoU
    """
    for dc_x0, dc_y0, dc_x1, dc_y1 in dontcare_boxes:
        iou = compute_iou((x0, y0, x1, y1), (dc_x0, dc_y0, dc_x1, dc_y1))
        print(f"🔍 IoU entre detección ({x0}, {y0}, {x1}, {y1}) y Don't Care ({dc_x0}, {dc_y0}, {dc_x1}, {dc_y1}) = {iou:.4f}")
        if iou > 0.5:
            return True
    return False

def non_max_suppression(boxes, scores, labels, iou_threshold=0.5):
    '''Filtrado de detecciones duplicadas con la técnica NMS'''
    if len(boxes) == 0:
        return [], [], []
    
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    indices = np.argsort(scores)[::-1] # Ordenar por score (descendente)

    keep = []
    while len(indices) > 0:
        i = indices[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[indices[1:]])
        yy1 = np.maximum(y1[i], y1[indices[1:]])
        xx2 = np.minimum(x2[i], x2[indices[1:]])
        yy2 = np.minimum(y2[i], y2[indices[1:]])

        inter_area = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2, yy1)
        iou = inter_area / (areas[i] + areas[indices[1:]] - inter_area)

        indices = indices[1:][iou < iou_threshold]

    return boxes[keep], scores[keep], labels[keep]

def generate_kitti_txt(image_path, model, output_txt_path):
    image, image_tensor = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(image_tensor)
    
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.8  # Umbral de confianza para el score
    bboxes = outputs['pred_boxes'][0, keep].numpy()
    labels = probas[keep].argmax(-1).cpu().numpy()
    scores = probas.max(-1).values[keep].cpu().numpy()

    width, height = image.size

    abs_bboxes = []
    for bbox in bboxes:
        cx, cy, w, h = bbox
        x0, y0 = int((cx - w / 2) * width), int((cy - h / 2) * height)
        x1, y1 = int((cx + w / 2 ) * width), int((cy + h / 2) * height)
        abs_bboxes.append([x0, y0, x1, y1])

    abs_bboxes = np.array(abs_bboxes)

    # Aplicar Non-Maximum Suppression (NMS)
    abs_bboxes, scores, labels = non_max_suppression(abs_bboxes, scores, labels, iou_threshold=0.5)
    dontcare_boxes = read_dontcare_regions('../labels/gt/000003.txt')
    with open(output_txt_path, 'w') as f:
        for bbox, label, score in zip(abs_bboxes, labels, scores):
            if label >= len(COCO_INSTANCE_CATEGORY_NAMES):
                print(f"⚠️ Advertencia: Se encontró un label fuera de rango ({label}), será ignorado.")
                continue  # Ignorar esta detección
            
            x0, y0, x1, y1 = bbox       
            line = f"{COCO_INSTANCE_CATEGORY_NAMES[label]} 0.00 0 0 {x0} {y0} {x1} {y1} 0 0 0 0 0 0 0 {score:.4f}\n"
            f.write(line)
    
    plot_detr_results(image, abs_bboxes, labels)  # Mostrar la imagen con los resultados

# Cargar el modelo DETR
detr_model = load_detr_model()

# Ruta de la imagen y del archivo de salida
image_folder = "../images/obj_det_images/"
output_folder = "../labels/pred/detr/"

# Obtener todas las imágenes en la carpeta
image_paths = glob.glob(os.path.join(image_folder, "*.png"))

for image_path in image_paths:
    image_name = os.path.basename(image_path)
    output_txt_path = os.path.join(output_folder, image_name.replace(".png", ".txt"))
    generate_kitti_txt(image_path, detr_model, output_txt_path)

generate_kitti_txt(image_path, detr_model, output_txt_path)