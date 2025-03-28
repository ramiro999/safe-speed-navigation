# image_processing.py

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms as T
from detr.model_detr import COCO_INSTANCE_CATEGORY_NAMES
import cv2

# Generar colores aleatorios para cada categoría
COLORS = np.random.rand(len(COCO_INSTANCE_CATEGORY_NAMES), 3) * 255

# Transformación para las imágenes de entrada
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def preprocess_image(image_input):
    if isinstance(image_input, str):
        image = Image.open(image_input)
    elif isinstance(image_input, np.ndarray):
        image = Image.fromarray(image_input)
    else:
        raise ValueError("Unsupported image input type")
    
    image_tensor = transform(image).unsqueeze(0)
    return image, image_tensor

def plot_detr_results(image, bboxes, labels):
    """
    Renderiza los resultados de detección con bounding boxes, IDs y nombres de las categorías.
    """

    if isinstance(image, str):
        image = Image.open(image)  # Cargar imagen si es una ruta

    if isinstance(image, Image.Image):
        image = np.array(image)

    # Asegurar que la imagen está en RGB antes de OpenCV
    if image.shape[-1] == 3:  
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Si no hay detecciones, mostrar advertencia
    if len(bboxes) == 0:
        print("⚠️ Advertencia: No hay detecciones para dibujar.")
        cv2.imshow('DETR', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return image

    # Convertir bounding boxes a coordenadas absolutas
    for bbox, label in zip(bboxes, labels):
        if label >= len(COCO_INSTANCE_CATEGORY_NAMES):
            print(f"⚠️ Advertencia: Se encontró un label fuera de rango ({label}), será ignorado.")
            continue

        x0, y0, x1, y1 = map(int, bbox)  # Asegurar coordenadas en enteros

        # Calcular el color para la categoría
        color = (int(COLORS[label][0]), int(COLORS[label][1]), int(COLORS[label][2]))

        # Dibujar la bounding box
        cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)

        # Mostrar ID del objeto y nombre de la categoría
        cv2.putText(image, f"category: {COCO_INSTANCE_CATEGORY_NAMES[label]}", 
                    (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    # Mostrar imagen con OpenCV
    cv2.imshow('DETR', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return image

