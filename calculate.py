import numpy as np
import math
import cv2
import os

# Leer archivo .npy y conocer sus valores

def read_npy_file(file_path="outputs/disparity_results/disp_pred.npy"):
    # Leer archivo .npy
    disp_pred = np.load(file_path)
    # Imprimir los valores mínimo y máximo
    print(f"Min: {disp_pred.min()}, Max: {disp_pred.max()}")
    # Imprimir la forma de la matriz
    print(f"Shape: {disp_pred.shape}")
    # Imprimir la matriz
    print(disp_pred)
    # Devolver la matriz
    return disp_pred

# Llamar a la funcion
# disp_pred = read_npy_file()



def read_ground_truth(file_path="ground_truth/depth_maps/0000000005.png"):
    # Leer archivo .npy
    ground_truth = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

    if ground_truth is None:
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Imprimir los valores mínimo y máximo
    print(f"Min: {ground_truth.min()}, Max: {ground_truth.max()}")
    # Imprimir la forma de la matriz
    print(f"Shape: {ground_truth.shape}")
    print(f'Tipo de dato: {ground_truth.dtype}')

    # Imprimir la matriz
    print(ground_truth)

    # Devolver la matriz
    return ground_truth

# Llamar a la funcion
ground_truth = read_ground_truth()

def read_image(file_path="outputs/depth.png"):
    # Leer archivo .npy
    image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

    if image is None:
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Imprimir los valores mínimo y máximo
    print(f"Min: {image.min()}, Max: {image.max()}")
    # Imprimir la forma de la matriz
    print(f"Shape: {image.shape}")
    print(f'Tipo de dato: {image.dtype}')

    # Imprimir la matriz
    print(image)

    # Devolver la matriz
    return image

# Llamar a la funcion
# image = read_image()

