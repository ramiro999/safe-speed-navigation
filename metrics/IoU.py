import numpy as np
import os
from datetime import datetime

def calculate_iou(box1, box2):
    """
    Calcula el Intersection over Union (IoU) entre dos *bounding boxes*.
    
    Args:
        box1, box2: Listas [x_min, y_min, x_max, y_max] de las cajas.

    Returns:
        IoU (float): Valor entre 0 y 1.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Área de la intersección
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Áreas individuales de las cajas
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Área de la unión
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0


def inter_model_agreement(detr_bboxes, yolo_bboxes, iou_threshold=0.5):
    """
    Calcula la coherencia entre las detecciones de DETR y YOLOv11.
    
    Args:
        detr_bboxes: Lista de *bounding boxes* de DETR ([x_min, y_min, x_max, y_max]).
        yolo_bboxes: Lista de *bounding boxes* de YOLOv11 ([x_min, y_min, x_max, y_max]).
        iou_threshold: Umbral de IoU para considerar una coincidencia válida.
    
    Returns:
        Dict con las métricas:
        - Tasa de coincidencia DETR→YOLO.
        - Tasa de coincidencia YOLO→DETR.
        - IoU promedio.
    """
    if not detr_bboxes or not yolo_bboxes:
        return {
            "DETR_to_YOLO_Match_Rate": 0.0,
            "YOLO_to_DETR_Match_Rate": 0.0,
            "Average_IoU": 0.0,
        }

    detr_to_yolo_matches = []
    yolo_to_detr_matches = []
    iou_values = []

    # Coincidencias DETR → YOLO
    for box_d in detr_bboxes:
        matched = False
        for box_y in yolo_bboxes:
            iou = calculate_iou(box_d, box_y)
            if iou >= iou_threshold:
                matched = True
                iou_values.append(iou)
                break
        detr_to_yolo_matches.append(matched)

    # Coincidencias YOLO → DETR
    for box_y in yolo_bboxes:
        matched = False
        for box_d in detr_bboxes:
            iou = calculate_iou(box_y, box_d)
            if iou >= iou_threshold:
                matched = True
                break
        yolo_to_detr_matches.append(matched)

    # Cálculo de métricas
    detr_match_rate = sum(detr_to_yolo_matches) / len(detr_bboxes) if detr_bboxes else 0
    yolo_match_rate = sum(yolo_to_detr_matches) / len(yolo_bboxes) if yolo_bboxes else 0
    avg_iou = np.mean(iou_values) if iou_values else 0

    return {
        "DETR_to_YOLO_Match_Rate": detr_match_rate,
        "YOLO_to_DETR_Match_Rate": yolo_match_rate,
        "Average_IoU": avg_iou
    }


def save_metrics_to_file(metrics, folder_path="./outputs"):
    """
    Guarda las métricas de inter-model agreement en un archivo .txt con timestamp.
    
    Args:
        metrics (dict): Diccionario con las métricas calculadas.
        folder_path (str): Carpeta donde se guardará el archivo.
    """
    os.makedirs(folder_path, exist_ok=True)  # Crear directorio si no existe

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(folder_path, f"inter_model_agreement_metrics_{timestamp}.txt")

    with open(file_path, "w") as file:
        file.write("Inter-Model Agreement Metrics\n")
        file.write("==============================\n")
        for key, value in metrics.items():
            file.write(f"{key}: {value:.4f}\n")

    print(f"Métricas guardadas en: {file_path}")
