import numpy as np

def calculate_d1_metric(D_est, D_gt, mask=None, threshold_abs=3, threshold_rel=0.05):
    """
    Calcula la métrica D1 para un disparity map estimado.

    Args:
        D_est (numpy.ndarray): Disparity map estimado.
        D_gt (numpy.ndarray): Ground truth del disparity map.
        mask (numpy.ndarray, opcional): Máscara binaria que define las regiones de interés (1=considerar, 0=ignorar).
        threshold_abs (float): Umbral absoluto (en píxeles).
        threshold_rel (float): Umbral relativo (proporción de la disparidad del ground truth).

    Returns:
        float: Porcentaje de píxeles incorrectos según la métrica D1.
    """
    # Asegurarse de que mask está definido
    if mask is None:
        mask = np.ones_like(D_gt, dtype=bool)  # Considerar toda la imagen
    
    # Filtrar valores válidos usando la máscara
    valid_pixels = mask & (D_gt > 0)  # Considerar solo píxeles válidos en el ground truth
    
    # Calcular el error absoluto entre D_est y D_gt
    error = np.abs(D_est - D_gt)
    
    # Calcular el umbral dinámico
    threshold = np.maximum(threshold_abs, threshold_rel * D_gt)
    
    # Identificar píxeles incorrectos
    incorrect_pixels = (error > threshold) & valid_pixels
    
    # Calcular porcentaje de píxeles incorrectos
    D1_error = np.sum(incorrect_pixels) / np.sum(valid_pixels) * 100  # Porcentaje
    
    return D1_error


# Ejemplo de disparity map y ground truth
D_est = np.array([[10, 15, 20], [30, 25, 0], [0, 40, 45]], dtype=float)
D_gt = np.array([[10, 15, 20], [30, 25, 5], [0, 40, 45]], dtype=float)

# Máscara para el fondo (bg)
bg_mask = (D_gt < 30)  # Define el fondo como disparidades menores a 30

# Máscara para toda la imagen
all_mask = np.ones_like(D_gt, dtype=bool)

# Cálculo de D1 para fondo (D1-bg) y toda la imagen (D1-all)
D1_bg = calculate_d1_metric(D_est, D_gt, mask=bg_mask)
D1_all = calculate_d1_metric(D_est, D_gt, mask=all_mask)

print(f"D1-bg: {D1_bg:.2f}%")
print(f"D1-all: {D1_all:.2f}%")
