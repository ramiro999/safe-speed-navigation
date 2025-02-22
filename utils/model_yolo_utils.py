import numpy as np
import cv2
import torch
from PIL import Image
import os
import sys
from ultralytics.utils.plotting import Annotator
from ultralytics import YOLO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from yolov11.image_processing_yolo import preprocess_image, draw_yolo_results
from yolov11.model_yolo import load_yolov11_model

"""
Este script obtiene el .txt con la información necesaria para evaluar la detección de objetos (IoU y mAP)
utilizando el modelo YOLOv11. Se extrae la clase del objeto, las coordenadas de la bbox y la confianza.
"""

# Mapeo de nombres de clases de COCO a nombres KITTI
CATEGORY_NAME_MAPPING = {
    "person": "Pedestrian",
    "bicycle": "Cyclist",
    "car": "Car",
    "truck": "Truck"
}

def generate_kitti_txt(image_path, model, output_txt_path, conf_threshold=0.5):
    """
    Genera un archivo de prediccion en formato KITTI

    Args:
        image_path (str): Ruta de la imagen de entrada
        model (YOLO): Modelo YOLOv11
        output_txt_path (str): Ruta de salida de la predicción
        conf_threshold (float): Umbral de confianza para filtrar detecciones
    """
    image = preprocess_image(image_path)

    results = model(image)

    width, height = image.size

    with open(output_txt_path, 'w') as f:
        for box in results[0].boxes:
            xyxy = box.xyxy[0].tolist() # [x_min, y_min, x_max, y_max]
            conf = box.conf[0].item() # Confianza
            cls = int(box.cls[0]) # Clase

            # Obtener nombre de clase y mapearlo si es necesario
            class_name = results[0].names[cls]
            if class_name not in CATEGORY_NAME_MAPPING:
                continue # Ignorar detecciones que no están en la lista

            mapped_class = CATEGORY_NAME_MAPPING[class_name]

            # Filtrar por umbral de confianza
            if conf < conf_threshold:
                continue

            # Escribir en formato KITTI
            x0, y0, x1, y1 = map(int, xyxy)
            line = f"{mapped_class} 0.00 0 0 {x0} {y0} {x1} {y1} 0 0 0 0 0 0 0 {conf:.4f}\n"
            f.write(line)

    annotated_img = draw_yolo_results(image_path, results[0])
    cv2.imshow("YOLOv11 Results", annotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Cargar el modelo YOLOv11
yolo_model_path = "./yolov11/model/yolo11s.pt"
yolo_model = load_yolov11_model(yolo_model_path)

# Ruta de la imagen y del archivo de salida
image_path = "../images/obj_det_images/000003.png"
output_txt_path = "../labels/pred/yolo/000003.txt"

# Generar el archivo KITTI
generate_kitti_txt(image_path, yolo_model, output_txt_path)