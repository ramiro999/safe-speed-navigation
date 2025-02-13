import os
import torch
import numpy as np
from shapely.geometry import box
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# Definir clases y niveles de dificultad
CLASSES = ['Car', 'Pedestrian', 'Cyclist']
DIFFICULTY = {'EASY': 0, 'MODERATE': 1, 'HARD': 2}

# Cargar anotaciones ground truth para múltiples archivos
def load_ground_truth(gt_folder):
    gt_data = {}
    for filename in os.listdir(gt_folder):
        if filename.endswith('.txt'):
            img_id = filename.split('.')[0]  # Extrae el identificador numérico
            with open(os.path.join(gt_folder, filename), 'r') as f:
                gt_data[img_id] = [line.strip().split() for line in f.readlines()]
    return gt_data

# Cargar predicciones para múltiples archivos
def load_predictions(pred_folder):
    pred_data = {}
    for filename in os.listdir(pred_folder):
        if filename.endswith('.txt'):
            img_id = filename.split('.')[0]  # Extrae el identificador numérico
            with open(os.path.join(pred_folder, filename), 'r') as f:
                pred_data[img_id] = [line.strip().split() for line in f.readlines()]
    return pred_data

# Calcular Intersection over Union (IoU)
def compute_iou(box1, box2):
    b1 = box(float(box1[0]), float(box1[1]), float(box1[2]), float(box1[3]))
    b2 = box(float(box2[0]), float(box2[1]), float(box2[2]), float(box2[3]))
    intersection = b1.intersection(b2).area
    union = b1.union(b2).area
    return intersection / union if union > 0 else 0

# Evaluación de detección 2D con PyTorch
def evaluate_2d_detection_pytorch(gt_folder, pred_folder, iou_threshold=0.5):
    gt_data = load_ground_truth(gt_folder)
    pred_data = load_predictions(pred_folder)
    
    metric = MeanAveragePrecision(iou_thresholds=[iou_threshold])
    
    for img_id in gt_data:
        gt_boxes = torch.tensor([[float(x) for x in obj[4:8]] for obj in gt_data[img_id]], dtype=torch.float32)
        gt_labels = torch.tensor([CLASSES.index(obj[0]) + 1 for obj in gt_data[img_id]], dtype=torch.int64)
        
        pred_boxes = torch.tensor([[float(x) for x in obj[4:8]] for obj in pred_data.get(img_id, [])], dtype=torch.float32)
        pred_scores = torch.tensor([float(obj[8]) for obj in pred_data.get(img_id, [])], dtype=torch.float32)
        pred_labels = torch.tensor([CLASSES.index(obj[0]) + 1 for obj in pred_data.get(img_id, [])], dtype=torch.int64)
        
        metric.update(
            preds=[{"boxes": pred_boxes, "scores": pred_scores, "labels": pred_labels}],
            target=[{"boxes": gt_boxes, "labels": gt_labels}]
        )
    
    results = metric.compute()
    return results

# Ejecutar evaluación
gt_folder = '../labels/gt/'  # Ruta a etiquetas reales
pred_folder = '../labels/pred/'  # Ruta a detecciones
results = evaluate_2d_detection_pytorch(gt_folder, pred_folder)
print("Evaluation Results:", results)
