import numpy as np
import cv2
from ultralytics.utils.plotting import Annotator  # Utilidad para dibujar
from PIL import Image

def preprocess_image(image_input):
    """
    Preprocesa una imagen para usarla con YOLO.
    """
    if isinstance(image_input, str):
        image = Image.open(image_input)
    elif isinstance(image_input, np.ndarray):
        image = Image.fromarray(image_input)
    else:
        raise ValueError("Unsupported image input type")
    return image

def draw_yolo_results(image_path, results):
    """
    Dibuja los resultados de YOLO en la imagen.

    Args:
        image_path (str): Ruta de la imagen de entrada.
        results: Resultados de YOLO.

    Returns:
        np.ndarray: Imagen con las detecciones dibujadas.
    """
    image = cv2.imread(image_path)  # Leer la imagen
    annotator = Annotator(image)  # Instanciar el anotador

    # Iterar sobre los resultados y dibujar las cajas
    for box in results.boxes:
        xyxy = box.xyxy[0].tolist()  # Coordenadas de la caja [x_min, y_min, x_max, y_max]
        conf = box.conf[0]  # Confianza
        cls = int(box.cls[0])  # Clase
        label = f"{results.names[cls]} {conf:.2f}"  # Etiqueta
        annotator.box_label(xyxy, label, color=(255, 0, 0))  # Dibujar

    return annotator.result()  # Imagen con anotaciones
