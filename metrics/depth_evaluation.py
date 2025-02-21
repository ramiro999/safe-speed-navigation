import torch
from PIL import Image
import numpy as np
import csv
import os
from glob import glob

# Clase para evaluar métricas
class MetricEvaluator:
    def __init__(self, metrics):
        self.metrics_fn = {metric: globals()[metric] for metric in metrics} 
        self.results = {metric: 0.0 for metric in metrics}

    def evaluate_metrics(self, pred, groundtruth):
        """Calcula las métricas de error entre la predicción y el ground truth."""

        # Convertir tensores a NumPy
        pred = pred.squeeze().numpy()
        groundtruth = groundtruth.squeeze().numpy()

        # Máscara para valores válidos
        valid_mask = (groundtruth > 1e-8)

        # Aplicar la máscara
        pred_valid = pred[valid_mask]
        groundtruth_valid = groundtruth[valid_mask]

        # Convertir unidades
        pred_mm = pred_valid * 1000.0
        groundtruth_mm = groundtruth_valid * 1000.0
        pred_km = pred_valid / 1000.0
        groundtruth_km = groundtruth_valid / 1000.0
        pred_inv_km = 1.0 / (pred_km + 1e-8) 
        groundtruth_inv_km = 1.0 / (groundtruth_km + 1e-8)

        # Calcular métricas
        for metric in self.results.keys():
            if metric in ['mae_metric', 'rmse_metric']:
                self.results[metric] = self.metrics_fn[metric](pred_mm, groundtruth_mm)
            elif metric in ['imae_metric', 'irmse_metric']:
                self.results[metric] = self.metrics_fn[metric](pred_inv_km, groundtruth_inv_km)

        return self.results.copy()

# Funciones de métricas
def mae_metric(pred_mm, groundtruth_mm):
    return np.abs(pred_mm - groundtruth_mm).mean()

def rmse_metric(pred_mm, groundtruth_mm):
    return np.sqrt(np.mean(np.square(pred_mm - groundtruth_mm)))

def imae_metric(pred_inv_km, groundtruth_inv_km):
    return np.abs(pred_inv_km - groundtruth_inv_km).mean()

def irmse_metric(pred_inv_km, groundtruth_inv_km):
    return np.sqrt(np.mean(np.square(pred_inv_km - groundtruth_inv_km)))

# Función para cargar una imagen
def load_image_as_tensor(image_path):
    """Carga una imagen de profundidad y la convierte en un tensor de PyTorch."""
    image = Image.open(image_path)
    image_array = np.array(image, dtype=np.uint16) / 256.0
    tensor = torch.tensor(image_array, dtype=torch.float32)
    return tensor

# Función para guardar resultados en CSV
def save_metrics_to_csv(csv_path, image_name, results):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["image_name", "mae", "rmse", "imae", "irmse"])  # Encabezados en el orden correcto
        writer.writerow([image_name, results["mae_metric"], results["rmse_metric"], results["imae_metric"], results["irmse_metric"]])

# Procesar todas las imágenes en una carpeta
def process_images(pred_dir, gt_dir, output_csv):
    pred_images = sorted(glob(os.path.join(pred_dir, "*.png")))
    gt_images = sorted(glob(os.path.join(gt_dir, "*.png")))

    if len(pred_images) == 0 or len(gt_images) == 0:
        print("No se encontraron imágenes en las carpetas proporcionadas.")
        return

    print(f"Procesando {len(pred_images)} imágenes...")

    # Definir métricas a calcular
    metrics = ['mae_metric', 'rmse_metric', 'imae_metric', 'irmse_metric']
    evaluator = MetricEvaluator(metrics)

    for pred_path, gt_path in zip(pred_images, gt_images):
        pred_tensor = load_image_as_tensor(pred_path)
        groundtruth_tensor = load_image_as_tensor(gt_path)

        results = evaluator.evaluate_metrics(pred_tensor, groundtruth_tensor)

        image_name = os.path.basename(pred_path)
        save_metrics_to_csv(output_csv, image_name, results)
        print(f"Procesado: {image_name}")

    print(f"\nTodas las métricas se han guardado en '{output_csv}'")

# Ruta de carpetas
pred_dir = "../outputs/depth_results"  # Carpeta con imágenes de predicción
gt_dir = "../ground_truth/depth_maps/kitti_2015/train/person"  # Carpeta con ground truths
output_csv = "metric_results_train.csv"

# Ejecutar procesamiento
if __name__ == "__main__":
    process_images(pred_dir, gt_dir, output_csv)
