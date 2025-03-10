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
    dontcare_data = {}

    for filename in os.listdir(gt_folder):
        if filename.endswith('.txt'):
            img_id = filename.split('.')[0].zfill(5)  # Extrae el identificador numérico con ceros a la izquierda
            with open(os.path.join(gt_folder, filename), 'r') as f:
                gt_objects = []
                dontcare_boxes = []

                for line in f.readlines():
                    values = line.split()
                    obj_class = values[0]
                    bbox = list(map(float, values[4:8]))  # bbox [x_min, y_min, x_max, y_max]

                    if obj_class in CLASSES:
                        gt_objects.append({"class": obj_class, "bbox": bbox})
                    elif obj_class == "DontCare":
                        dontcare_boxes.append(bbox)

                gt_data[img_id] = gt_objects
                dontcare_data[img_id] = dontcare_boxes  # Almacenar regiones "Don't Care"

    return gt_data, dontcare_data

# Cargar predicciones para múltiples archivos
def load_predictions(pred_folder):
    pred_data = {}

    for filename in os.listdir(pred_folder):
        if filename.endswith('.txt'):
            img_id = filename.split('.')[0].zfill(5)
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
    b1 = box(*box1)
    b2 = box(*box2)
    intersection = b1.intersection(b2).area
    union = b1.union(b2).area
    return intersection / union if union > 0 else 0

# Filtrar detecciones en "Don't Care"
def filter_dontcare_detections(pred_data, dontcare_data, iou_threshold=0.5):
    '''
    En caso de tener regionres Don't Care, filtra las detecciones que se encuentren dentro de estas regiones (coordenadas de las cajas).
    Es decir, computa las etiquetas del gt y del pred, donde se busca la que tenga el IoU mayor para esa etiqueta del GT. 
    Se ignora esa detección si el IoU es mayor al umbral establecido.

    Args:
        pred_data (dict): Diccionario con las predicciones.
        dontcare_data (dict): Diccionario con las regiones "Don't Care".
        iou_threshold (float): Umbral de IoU para ignorar detecciones.

    Returns:
        filtered_preds (dict): Diccionario con las predicciones filtradas.
    '''

    filtered_preds = {}

    for img_id, predictions in pred_data.items():
        filtered_preds[img_id] = []
        dontcare_boxes = dontcare_data.get(img_id, [])

        for pred in predictions:
            x0, y0, x1, y1 = pred["bbox"]
            ignore_detection = False

            for dc_box in dontcare_boxes:
                iou = compute_iou((x0, y0, x1, y1), dc_box)
                if iou > iou_threshold:
                    ignore_detection = True
                    print(f"🔹 Detección en región 'Don't Care' ignorada: {pred['class']} con IoU={iou:.2f}")
                    break

            if not ignore_detection:
                filtered_preds[img_id].append(pred)

    return filtered_preds

# Calcular IoU por imagen y promedio
def compute_iou_per_image(gt_data, pred_data):
    '''
    Computa las etiquetas de la verdad fundamental sobre cada prediccion buscando la que tenga el IoU mayor para esa etiqueta del GT.

    Args:
        gt_data (dict): Diccionario con las anotaciones de la verdad fundamental.
        pred_data (dict): Diccionario con las predicciones.

    Returns:
        iou_results (dict): Diccionario con el IoU promedio por imagen.
        avg_iou (float): IoU promedio de todas las imágenes.
    '''
    iou_results = {}
    total_iou = []
    
    for img_id in gt_data: # Itera sobre las imágenes
        gt_boxes = [obj["bbox"] for obj in gt_data[img_id]]
        pred_boxes = [obj["bbox"] for obj in pred_data.get(img_id, [])]

        img_iou_values = []
        for gt_box in gt_boxes: # Itera sobre las detecciones reales 
            for pred_box in pred_boxes: # Itera sobre las detecciones predichas
                iou = compute_iou(gt_box, pred_box) 
                img_iou_values.append(iou) 
                print(f"📌 Imagen {img_id} - IoU entre GT {gt_box} y Predicción {pred_box}: {iou:.4f}")

        # Guardar el IoU promedio de la imagen
        if img_iou_values:
            iou_results[img_id] = sum(img_iou_values) / len(img_iou_values)
            total_iou.extend(img_iou_values)

    # Calcular el IoU promedio de todas las imágenes
    avg_iou = sum(total_iou) / len(total_iou) if total_iou else 0
    return iou_results, avg_iou


# Evaluación de detección 2D con PyTorch
def evaluate_2d_detection_pytorch(gt_folder, pred_folder, iou_threshold=0.5):
    gt_data, dontcare_data = load_ground_truth(gt_folder)  # Carga el GT y "DontCare"
    pred_data = load_predictions(pred_folder)  # Carga predicciones

    # Filtrar detecciones en "Don't Care"
    filtered_preds = filter_dontcare_detections(pred_data, dontcare_data, iou_threshold)

    metric = MeanAveragePrecision(iou_thresholds=[iou_threshold])

    for img_id in gt_data:
        # Ground Truth
        gt_boxes = torch.tensor([obj["bbox"] for obj in gt_data[img_id]], dtype=torch.float32)
        gt_labels = torch.tensor([CLASSES.index(obj["class"]) + 1 for obj in gt_data[img_id]], dtype=torch.int64)

        # Predicciones después de filtrar "Don't Care"
        pred_boxes = torch.tensor([obj["bbox"] for obj in filtered_preds.get(img_id, [])], dtype=torch.float32)
        pred_scores = torch.tensor([obj["score"] for obj in filtered_preds.get(img_id, [])], dtype=torch.float32)
        pred_labels = torch.tensor([CLASSES.index(obj["class"]) + 1 for obj in filtered_preds.get(img_id, [])], dtype=torch.int64)

        # Actualizar métrica
        metric.update(
            preds=[{"boxes": pred_boxes, "scores": pred_scores, "labels": pred_labels}],
            target=[{"boxes": gt_boxes, "labels": gt_labels}]
        )

    results = metric.compute()

    # Calcular IoU por imagen y promedio
    iou_per_image, average_iou = compute_iou_per_image(gt_data, filtered_preds)
    
    results["average_iou"] = average_iou
    results["iou_per_image"] = iou_per_image

    return results

# Ejecutar evaluación
if __name__ == "__main__":
    gt_folder = '../labels/gt/'  # Ruta a etiquetas reales
    pred_folder = '../labels/pred/yolo/'  # Ruta a detecciones

    # Evaluar
    results = evaluate_2d_detection_pytorch(gt_folder, pred_folder)
    results_clean = {k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in results.items()}
    
    # Imprimir resultados
    
    print(json.dumps(results_clean, indent=4))
    print("Evaluation Results:", results)
