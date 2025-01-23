import numpy as np
import cv2
from sklearn.cluster import KMeans

def disp_read(filename):
    """Load disparity map from PNG file."""
    I = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    #if len(I.shape) == 3 and I.shape[-1] > 1:
        #I = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY).astype(np.float32)
        
    D = I.astype(np.float32) / 256.0
    #D[I == 0] = -1  # Set invalid pixels
    return D


def calculate_mse(D_gt, D_est):
    """
    Calcula el Error Cuadrático Medio (MSE) entre la verdad de terreno (D_gt)
    y la disparidad estimada (D_est).
    
    Parámetros:
    - D_gt: numpy array de la verdad de terreno.
    - D_est: numpy array de la disparidad estimada.
    
    Retorna:
    - mse: valor del error cuadrático medio.
    """
    mask_valid = (D_gt > 0)  # Considerar solo valores válidos (evitar píxeles inválidos)
    mse = np.mean((D_gt[mask_valid] - D_est[mask_valid]) ** 2)
    return mse

def calculate_rmse(D_gt, D_est):
    """
    Calcula la Raíz del Error Cuadrático Medio (RMSE) entre la verdad de terreno (D_gt)
    y la disparidad estimada (D_est).
    
    Parámetros:
    - D_gt: numpy array de la verdad de terreno.
    - D_est: numpy array de la disparidad estimada.
    
    Retorna:
    - rmse: valor de la raíz del error cuadrático medio.
    """
    mask_valid = (D_gt > 0)  # Considerar solo valores válidos (evitar píxeles inválidos)
    mse = np.mean((D_gt[mask_valid] - D_est[mask_valid]) ** 2)
    rmse = np.sqrt(mse)
    return rmse


def calculate_sze(D_gt, D_est, f, B, mu=1.0):
    """
    Calcula el error SZE (Sum of Z-normalized Errors) entre la verdad de terreno (D_gt)
    y la disparidad estimada (D_est).
    
    Parámetros:
    - D_gt: numpy array de la verdad de terreno.
    - D_est: numpy array de la disparidad estimada.
    - f: longitud focal de la cámara.
    - B: distancia entre las cámaras (baseline).
    - mu: parámetro de regularización para evitar divisiones por cero.
    
    Retorna:
    - sze: valor del error SZE.
    """
    mask_valid = (D_gt > 0.0)  # Considerar solo valores válidos
    term_gt = (f * B) / (D_gt[mask_valid] + mu)
    term_est = (f * B) / (D_est[mask_valid] + mu)
    sze = np.sum(np.abs(term_gt - term_est))
    return sze

def calculate_bmp(D_gt, D_est, delta):
    """
    Calcula la métrica BMP (Bad Matching Pixels), que mide el porcentaje de píxeles 
    cuya diferencia supera un umbral delta.
    
    Parámetros:
    - D_gt: numpy array de la verdad de terreno (ground truth).
    - D_est: numpy array de la disparidad estimada.
    - delta: umbral de error aceptable.
    
    Retorna:
    - bmp: porcentaje de píxeles con error mayor a delta.
    """
    mask_valid = (D_gt > 0)  # Considerar solo valores válidos
    error_pixels = np.abs(D_gt[mask_valid] - D_est[mask_valid]) > delta
    bmp = np.sum(error_pixels) / np.sum(mask_valid)
    return bmp * 100  # Convertir a porcentaje

def calculate_mre(D_gt, D_est):
    """
    Calcula el Error Relativo Medio (MRE - Mean Relative Error).
    
    Parámetros:
    - D_gt: numpy array de la verdad de terreno.
    - D_est: numpy array de la disparidad estimada.
    
    Retorna:
    - mre: valor del error relativo medio.
    """
    mask_valid = (D_gt > 0.0)  # Evitar divisiones por cero
    relative_error = np.abs(D_gt[mask_valid] - D_est[mask_valid]) / D_gt[mask_valid]
    mre = np.mean(relative_error)
    return mre * 100  # Convertir a porcentaje

def calculate_bmpre(D_gt, D_est, delta, delta_prime):
    """
    Calcula la métrica BMPRE (Bad Matching Pixels Relative Error), que mide el porcentaje de errores 
    ponderado por el error relativo.

    Parámetros:
    - D_gt: numpy array de la verdad de terreno.
    - D_est: numpy array de la disparidad estimada.
    - delta: umbral para considerar un error significativo.
    - delta_prime: umbral más relajado para errores menores.

    Retorna:
    - bmpre: métrica BMPRE calculada.
    """
    mask_valid = (D_gt > 0)
    delta_error = np.abs(D_gt[mask_valid] - D_est[mask_valid])
    relative_error = delta_error / D_gt[mask_valid]

    # Aplicar la función tau según la condición dada en la ecuación
    tau = np.where(D_gt[mask_valid] > 0, relative_error, 0)

    # Contar píxeles que superan el umbral delta
    bmpre = np.sum(tau[delta_error > delta]) / np.sum(delta_error > delta_prime)
    return bmpre * 100  # Convertir a porcentaje

# Ejemplo de uso con valores ficticios
D_gt = disp_read("../ground_truth/disparity_maps/000013_10.png")
D_est = disp_read("../outputs/disparity_results/000013_10.png")

# Normalizar la disparidad estimada
# D_est = (D_est / np.max(D_est)) * np.max(D_gt[D_gt > 0]) 


f = 725.0087  # Ejemplo de longitud focal
B = 0.532725 # Ejemplo de baseline


delta = 3  # Umbral para BMP
delta_prime = 1  # Umbral para BMPRE

bmp_value = calculate_bmp(D_gt, D_est, delta)
mre_value = calculate_mre(D_gt, D_est)
bmpre_value = calculate_bmpre(D_gt, D_est, delta, delta_prime)
mse_value = calculate_mse(D_gt, D_est)
sze_value = calculate_sze(D_gt, D_est, f, B)
rmse_value = calculate_rmse(D_gt, D_est)

print(f"MSE: {mse_value:.5f}")
print(f"SZE: {sze_value:.5f}")
print(f"BMP: {bmp_value:.5f}%")
print(f"MRE: {mre_value:.2f}%")
print(f"BMPRE: {bmpre_value:.5f}%")
print(f"RMSE: {rmse_value:.5f}")


# Mostrar las imágenes usando OpenCV
cv2.imshow('D_est', D_est)
cv2.imshow('D_gt', D_gt)

cv2.waitKey(0)
cv2.destroyAllWindows()

##################### -------------------------- #####################

# Mostrar los valores de D_gt y D_est
print("Valores de D_gt:")
print(D_gt)

print("Valores de D_est:")
print(D_est)

# Calcular y mostrar las diferencias
diferencias = D_gt - D_est
print("Diferencias entre D_gt y D_est:")
print(diferencias)

##################### -------------------------- #####################

import matplotlib.pyplot as plt

# Generar histograma de los valores de D_gt
plt.hist(D_gt.flatten(), bins=50, color='blue', alpha=0.7)
plt.title('Histograma de los valores de D_gt')
plt.xlabel('Valor de disparidad')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.savefig('/home/ramiro-avila/safe-speed-navigation/metrics/D_gt_histograma.png')
plt.clf()  # Limpiar la figura para el siguiente histograma

# Generar histograma de los valores de D_est
plt.hist(D_est.flatten(), bins=50, color='green', alpha=0.7)
plt.title('Histograma de los valores de D_est')
plt.xlabel('Valor de disparidad')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.savefig('/home/ramiro-avila/safe-speed-navigation/metrics/D_est_histograma.png')


# Generar gráfico de dispersión (scatter plot) de D_gt vs D_est
plt.scatter(D_gt.flatten(), D_est.flatten(), alpha=0.5, color='purple')
plt.title('Gráfico de dispersión de D_gt vs D_est')
plt.xlabel('D_gt')
plt.ylabel('D_est')
plt.grid(True)
plt.savefig('/home/ramiro-avila/safe-speed-navigation/metrics/D_gt_vs_D_est_scatter.png')
plt.clf()  # Limpiar la figura para el siguiente gráfico

# Generar gráfico de líneas para comparar D_gt y D_est
plt.plot(D_gt.flatten(), label='D_gt', color='blue', alpha=0.7)
plt.plot(D_est.flatten(), label='D_est', color='green', alpha=0.7)
plt.title('Gráfico de líneas de D_gt y D_est')
plt.xlabel('Índice de píxel')
plt.ylabel('Valor de disparidad')
plt.legend()
plt.grid(True)
plt.savefig('/home/ramiro-avila/safe-speed-navigation/metrics/D_gt_vs_D_est_line_plot.png')

# Calcular y mostrar el valor mínimo y máximo de D_gt
min_D_gt = np.min(D_gt[D_gt > 0])  # Ignorar valores inválidos
max_D_gt = np.max(D_gt)
print(f"Valor mínimo de D_gt: {min_D_gt}")
print(f"Valor máximo de D_gt: {max_D_gt}")

# Calcular y mostrar el valor mínimo y máximo de D_est
min_D_est = np.min(D_est[D_est > 0])  # Ignorar valores inválidos
max_D_est = np.max(D_est)
print(f"Valor mínimo de D_est: {min_D_est}")
print(f"Valor máximo de D_est: {max_D_est}")

# Acceder a los primeros 100 píxeles de cada imagen y crear listas con sus valores
D_gt_values = D_gt.flatten()[:1000]
D_est_values = D_est.flatten()[:1000]

# Mostrar los valores de los primeros 100 píxeles
print("Primeros 1000 valores de D_gt:")
print(D_gt_values)

print("Primeros 1000 valores de D_est:")
print(D_est_values)

##################### -------------------------- #####################