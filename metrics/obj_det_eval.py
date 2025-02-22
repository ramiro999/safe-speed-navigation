import os
import torch
import json
from shapely.geometry import box
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# Definir clases
CLASSES = ['Car', 'Pedestrian', 'Cyclist']

# Cargar anotaciones ground truth para múltiples archivos
def load_ground_truth(gt_folder):
    gt_data = {}
    for filename in os.listdir(gt_folder):
        if filename.endswith('.txt'):
            img_id = filename.split('.')[0]  # Extrae el identificador numérico
            with open(os.path.join(gt_folder, filename), 'r') as f:
                gt_data[img_id] = [
                    {
                        "class": line.split()[0],  # Tipo (clase)
                        "bbox": list(map(float, line.split()[4:8]))  # bbox [x_min, y_min, x_max, y_max]
                    }
                    for line in f.readlines() if line.split()[0] in CLASSES
                ]
    return gt_data

# Cargar predicciones para múltiples archivos
def load_predictions(pred_folder):
    pred_data = {}
    for filename in os.listdir(pred_folder):
        if filename.endswith('.txt'):
            img_id = filename.split('.')[0]  # Extrae el identificador numérico
            with open(os.path.join(pred_folder, filename), 'r') as f:
                pred_data[img_id] = [
                    {
                        "class": line.split()[0],  # Tipo (clase)
                        "bbox": list(map(float, line.split()[4:8])),  # bbox [x_min, y_min, x_max, y_max]
                        "score": float(line.split()[8])  # Puntuación de confianza
                    }
                    for line in f.readlines() if line.split()[0] in CLASSES
                ]
    return pred_data

# Calcular Intersection over Union (IoU)
def compute_iou(box1, box2):
    b1 = box(*box1)  # Crear un objeto de tipo box con las coordenadas
    b2 = box(*box2)
    intersection = b1.intersection(b2).area
    union = b1.union(b2).area
    return intersection / union if union > 0 else 0

def compute_average_iou(gt_data, pred_data):
    iou_values = []

    for img_id in gt_data:
        gt_boxes = [obj["bbox"] for obj in gt_data[img_id]]
        pred_boxes = [obj["bbox"] for obj in pred_data.get(img_id, [])]

        for gt_box in gt_boxes:
            for pred_box in pred_boxes:
                iou = compute_iou(gt_box, pred_box)
                iou_values.append(iou)
    
    return sum(iou_values) / len(iou_values) if iou_values else 0 # Se evita la division por 0

# Evaluación de detección 2D con PyTorch
def evaluate_2d_detection_pytorch(gt_folder, pred_folder, iou_threshold=0.5):
    gt_data = load_ground_truth(gt_folder)
    pred_data = load_predictions(pred_folder)
    
    metric = MeanAveragePrecision(iou_thresholds=[iou_threshold])
    
    for img_id in gt_data:
        # Ground Truth
        gt_boxes = torch.tensor([obj["bbox"] for obj in gt_data[img_id]], dtype=torch.float32)
        gt_labels = torch.tensor([CLASSES.index(obj["class"]) + 1 for obj in gt_data[img_id]], dtype=torch.int64)

        
        # Predicciones
        pred_boxes = torch.tensor([obj["bbox"] for obj in pred_data.get(img_id, [])], dtype=torch.float32)
        pred_scores = torch.tensor([obj["score"] for obj in pred_data.get(img_id, [])], dtype=torch.float32)
        pred_labels = torch.tensor([CLASSES.index(obj["class"]) + 1 for obj in pred_data.get(img_id, [])], dtype=torch.int64)
        
        # Actualizar métrica
        metric.update(
            preds=[{"boxes": pred_boxes, "scores": pred_scores, "labels": pred_labels}],
            target=[{"boxes": gt_boxes, "labels": gt_labels}]
        )
    
    results = metric.compute()

    # Calcular iou promedio
    average_iou = compute_average_iou(gt_data, pred_data)
    results["average_iou"] = average_iou

    return results

# Ejecutar evaluación
if __name__ == "__main__":
    gt_folder = '../labels/gt/'  # Ruta a etiquetas reales
    pred_folder = '../labels/pred/'  # Ruta a detecciones
    
    # Evaluar
    results = evaluate_2d_detection_pytorch(gt_folder, pred_folder)
    results_clean = {k: v.tolist() if isinstance(v, torch.Tensor) else v for k,v in results.items()}
    print(json.dumps(results_clean, indent=4))
    print("Evaluation Results:", results)