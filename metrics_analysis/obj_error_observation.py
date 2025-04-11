import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box
import pandas as pd
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# Directorios de las anotaciones
gt_folder = "../labels/gt/"
pred_folder = "../labels/pred/rt-detr/"
img_folder = "../images/obj_det_images/"

# Parámetros
iou_threshold = 0.5
dontcare_threshold = 0.5  # Umbral de IoU para ignorar detecciones en regiones DontCare
min_bbox_height = 25  # Altura mínima de una caja delimitadora en pixeles
map_metric = MeanAveragePrecision()

# Clases que nos interesan
target_classes = {"Car", "Pedestrian", "Cyclist"}

# Función para leer anotaciones desde un archivo TXT
def read_annotations(txt_file):
    boxes = []
    dontcare_boxes = []
    if not os.path.exists(txt_file):
        return boxes, dontcare_boxes
    
    with open(txt_file, 'r') as f:
        for line in f.readlines():
            values = line.split()
            label = values[0]
            x_min, y_min, x_max, y_max = map(float, values[4:8])
            height = y_max - y_min
            if label == "DontCare":
                dontcare_boxes.append([x_min, y_min, x_max, y_max])
            elif label in target_classes:  # Solo guardamos clases de interés
                boxes.append((label, [x_min, y_min, x_max, y_max]))
    
    return boxes, dontcare_boxes

# Función para calcular IoU
def compute_iou(box1, box2):
    b1 = box(*box1)
    b2 = box(*box2)
    intersection = b1.intersection(b2).area
    union = b1.union(b2).area
    return intersection / union if union > 0 else 0

def update_map_metric(iou_df):
    """
    Actualiza la métrica Mean Average Precision (mAP) con los datos de detecciones.
    """
    preds = []
    targets = []
    
    # Agrupar predicciones y ground truths
    image_ids = iou_df["Image ID"].unique()
    for image_id in image_ids:
        image_detections = iou_df[iou_df["Image ID"] == image_id]
        
        pred_boxes = []
        pred_scores = []
        pred_labels = []
        gt_boxes = []
        gt_labels = []
        
        for _, row in image_detections.iterrows():
            if row["Best Pred Box"] is not None:
                pred_boxes.append(row["Best Pred Box"])
                pred_scores.append(row["IoU"])  # IoU como proxy de score
                pred_labels.append(1)  # Se asume una sola clase por ahora
            
            if row["GT Box"] is not None:
                gt_boxes.append(row["GT Box"])
                gt_labels.append(1)
        
        preds.append({
            "boxes": torch.tensor(pred_boxes, dtype=torch.float) if pred_boxes else torch.empty((0, 4)),
            "scores": torch.tensor(pred_scores, dtype=torch.float) if pred_scores else torch.empty((0,)),
            "labels": torch.tensor(pred_labels, dtype=torch.int) if pred_labels else torch.empty((0,), dtype=torch.int)
        })
        
        targets.append({
            "boxes": torch.tensor(gt_boxes, dtype=torch.float) if gt_boxes else torch.empty((0, 4)),
            "labels": torch.tensor(gt_labels, dtype=torch.int) if gt_labels else torch.empty((0,), dtype=torch.int)
        })
    
    # Calcular métricas con torchmetrics
    map_metric.update(preds, targets)
    return map_metric.compute()

# Obtener la lista de archivos en la carpeta de ground truth
gt_files = sorted([f for f in os.listdir(gt_folder) if f.endswith('.txt')])

# Inicializar métricas globales
global_tp, global_fp, global_fn = 0, 0, 0
image_results = []

for gt_file in gt_files:
    image_id = gt_file.replace('.txt', '')  # Extraer el ID de la imagen
    gt_path = os.path.join(gt_folder, gt_file)
    pred_path = os.path.join(pred_folder, gt_file)
    img_path = os.path.join(img_folder, f"{image_id}.png")

    # Leer las anotaciones
    gt_boxes, dontcare_boxes = read_annotations(gt_path)
    pred_boxes, _ = read_annotations(pred_path)  # No necesitamos dontcare en pred

    # Filtrar detecciones en regiones DontCare
    filtered_preds = []
    for label, pred_box in pred_boxes:
        ignore = False
        for dc_box in dontcare_boxes:
            iou = compute_iou(pred_box, dc_box)
            if iou > dontcare_threshold:
                ignore = True
                print(f"🔹 Imagen {image_id} - Detección en 'DontCare' ignorada: {label} con IoU={iou:.2f}")
                break
        if not ignore:
            filtered_preds.append((label, pred_box))

    # Evaluación de TP, FP y FN
    tp, fp, fn = 0, 0, 0
    ious = []
    detected_gts = set()
    used_preds = set()

    # Buscar Verdaderos Positivos (TP) y Falsos Negativos (FN)
    for gt_label, gt_box in gt_boxes:
        best_iou = 0
        best_pred_idx = None
        best_pred_box = None

        for pred_idx, (pred_label, pred_box) in enumerate(filtered_preds):
            iou = compute_iou(gt_box, pred_box)
            if iou > best_iou:
                best_iou = iou
                best_pred_idx = pred_idx
                best_pred_box = pred_box

        if best_iou >= iou_threshold:
            tp += 1
            detected_gts.add(tuple(gt_box))
            used_preds.add(best_pred_idx)
            ious.append((image_id, gt_label, gt_box, best_pred_box, best_iou, "TP"))
        else:
            fn += 1
            ious.append((image_id, gt_label, gt_box, None, 0, "FN"))

    # Buscar Falsos Positivos (FP)
    for pred_idx, (pred_label, pred_box) in enumerate(filtered_preds):
        if pred_idx not in used_preds:
            fp += 1
            ious.append((image_id, pred_label, None, pred_box, 0, "FP"))

    # Guardar métricas globales
    global_tp += tp
    global_fp += fp
    global_fn += fn

    # Guardar resultados por imagen
    image_results.extend(ious)

# Convertir a DataFrame y calcular métricas globales
iou_df = pd.DataFrame(image_results, columns=["Image ID", "GT Label", "GT Box", "Best Pred Box", "IoU", "Status"])

# Cálculo de métricas globales
precision = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0
recall = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0
f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
mean_iou = iou_df["IoU"].mean()

# Calcular promedio de IoU por clase
# IoU para Car
iou_car = iou_df[iou_df["GT Label"] == "Car"]["IoU"]
mean_iou_car = iou_car.mean() if not iou_car.empty else 0
# IoU para Pedestrian
iou_ped = iou_df[iou_df["GT Label"] == "Pedestrian"]["IoU"]
mean_iou_ped = iou_ped.mean() if not iou_ped.empty else 0
# IoU para Cyclist
iou_cyc = iou_df[iou_df["GT Label"] == "Cyclist"]["IoU"]
mean_iou_cyc = iou_cyc.mean() if not iou_cyc.empty else 0

# Calcular mAP con torchmetrics
map_results = update_map_metric(iou_df)

print("\n📊 Métricas Globales de Evaluación")
print(f"✔ Verdaderos Positivos (TP): {global_tp}")
print(f"❌ Falsos Positivos (FP): {global_fp}")
print(f"⚠ Falsos Negativos (FN): {global_fn}")
print(f"📈 Precisión: {precision:.4f}")
print(f"📉 Recall: {recall:.4f}")
print(f"🎯 F1-Score: {f1_score:.4f}")
print(f"Promedio de IoU: {mean_iou:.4f}")
print(f"Promedio de IoU para Car: {mean_iou_car:.4f}")
print(f"Promedio de IoU para Pedestrian: {mean_iou_ped:.4f}")
print(f"Promedio de IoU para Cyclist: {mean_iou_cyc:.4f}")
print(f"🔹 mAP: {map_results['map']:.4f}")


# Guardar resultados en un archivo CSV
iou_df.to_csv("detections_evaluation_filtered.csv", index=False)
print("\n✅ Resultados guardados en 'detections_evaluation_filtered.csv'")