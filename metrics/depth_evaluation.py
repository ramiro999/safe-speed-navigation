# Librerias requeridas
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

# Clase para evaluar métricas
class MetricEvaluator:
    def __init__(self, metrics):
        self.metrics_fn = {metric: globals()[metric] for metric in metrics} 
        self.results = {metric: 0.0 for metric in metrics}

    def evaluate_metrics(self, pred, groundtruth):
        """Calcula las métricas de error entre la predicción y el ground truth."""

        # Convertir tensores a NumPy
        pred = pred.squeeze().numpy()  # Quitar dimensiones extra
        groundtruth = groundtruth.squeeze().numpy()

        # Máscara para valores válidos
        valid_mask = (groundtruth > 1e-8)

        # Aplicar la máscara para valores válidos
        pred_valid = pred[valid_mask]
        groundtruth_valid = groundtruth[valid_mask]

        # Convertir a milímetros
        pred_mm = pred_valid * 1000.0
        groundtruth_mm = groundtruth_valid * 1000.0

        # Convertir a kilómetros
        pred_km = pred_valid / 1000.0
        groundtruth_km = groundtruth_valid / 1000.0

        # Calcular el inverso en 1/km
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

# Función para cargar una imagen y convertirla a tensor con conversión adecuada
def load_image_as_tensor(image_path):
    """Carga una imagen de profundidad y la convierte en un tensor de PyTorch con la conversión adecuada."""

    # Cargar la imagen
    image = Image.open(image_path)

    # Convertir a numpy array
    image_array = np.array(image, dtype=np.uint16) / 256.0 
    # print(f"Min: {image_array.min()}, Max: {image_array.max()}")

    # Convertir a uint16 (escalado de 8-bit a 16-bit)
    # image_uint16 = image_array.astype(np.uint16) * 256

    # Convertir a tensor de PyTorch
    tensor = torch.tensor(image_array, dtype=torch.float32)
    # print(f"Min: {tensor.min().item()}, Max: {tensor.max().item()}")

    # Convertir a float dividiendo entre 256 para obtener metros
    # tensor = tensor.float() / 256.0
    
    return tensor  # Agregar dimensión de canal

# Ejemplo de uso
if __name__ == "__main__":
    pred_image_path = "../outputs/depth.png"
    groundtruth_image_path = "../ground_truth/depth_maps/0000000045.png"

    # Cargar las imágenes como tensores
    pred_tensor = load_image_as_tensor(pred_image_path)
    groundtruth_tensor = load_image_as_tensor(groundtruth_image_path)

    print(f"Predicción - Tipo: {pred_tensor.dtype}, Rango: {pred_tensor.min().item()} - {pred_tensor.max().item()}, Forma: {pred_tensor.shape}")
    print(f"Ground Truth - Tipo: {groundtruth_tensor.dtype}, Rango: {groundtruth_tensor.min().item()} - {groundtruth_tensor.max().item()}, Forma: {groundtruth_tensor.shape}")

    # Definir métricas a calcular
    metrics = ['mae_metric', 'rmse_metric', 'imae_metric', 'irmse_metric']

    # Crear el evaluador de métricas y calcular resultados
    evaluator = MetricEvaluator(metrics)
    results = evaluator.evaluate_metrics(pred_tensor, groundtruth_tensor)

    # Mostrar resultados
    print("\nResultados de las métricas:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f} ")