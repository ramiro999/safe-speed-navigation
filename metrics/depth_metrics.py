from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Cargar la imagen de ground truth (LiDAR) y la imagen estimada
image_path_gt = "../ground_truth/disparity_maps/000013_10.png"  # Sustituir con la ruta correcta de la imagen de LiDAR
image_path_est = "../outputs/disparity_results/000013_10.png"  # Sustituir con la ruta correcta de la imagen estimada

# Cargar las imágenes
img_gt = Image.open(image_path_gt)
img_est = Image.open(image_path_est)

# Convertir a escala de grises para facilitar la comparación
img_gt_gray = img_gt.convert('L')
img_est_gray = img_est.convert('L')

# Convertir a arrays de numpy
img_gt_array = np.array(img_gt_gray)
img_est_array = np.array(img_est_gray)

# Normalizar los valores de las imágenes (rango [0, 1])
img_gt_normalized = (img_gt_array - np.min(img_gt_array)) / (np.max(img_gt_array) - np.min(img_gt_array))
img_est_normalized = (img_est_array - np.min(img_est_array)) / (np.max(img_est_array) - np.min(img_est_array))

# Calcular el RMSE (Root Mean Square Error)
rmse = np.sqrt(np.mean((img_gt_normalized - img_est_normalized) ** 2))


# Calcular el MAE (Mean Absolute Error)
mae = np.mean(np.abs(img_gt_normalized - img_est_normalized))

# Mostrar el RMSE
print(f"RMSE entre la imagen de ground truth y la imagen estimada: {rmse}")
print(f"MAE entre la imagen de ground truth y la imagen estimada: {mae}")


# Mostrar las imágenes para inspección visual usando OpenCV
cv2.imshow("Ground Truth (Normalizada)", img_gt_normalized)
cv2.imshow("Imagen Estimada (Normalizada)", img_est_normalized)

# Esperar a que se presione una tecla y luego cerrar las ventanas
cv2.waitKey(0)
cv2.destroyAllWindows()
