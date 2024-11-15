# image_processing.py

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms as T
from detr.model_loader import COCO_INSTANCE_CATEGORY_NAMES

# Generar colores aleatorios para las categorías
COLORS = np.random.rand(len(COCO_INSTANCE_CATEGORY_NAMES), 3)

# Transformación para las imágenes de entrada
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def preprocess_image(image_path):
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)
    return image, image_tensor

def plot_detr_results(image, bboxes, labels, ids):
    """
    Renderiza los resultados de detección con bounding boxes, IDs y nombres de las categorías.

    Args:
        image: Imagen de entrada en formato PIL.
        bboxes: Lista de bounding boxes normalizados ([cx, cy, w, h]).
        labels: Lista de etiquetas de categorías detectadas.
        ids: Lista de IDs únicos de los objetos detectados.

    Returns:
        fig: Figura de Matplotlib con los resultados renderizados.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)

    for bbox, label, obj_id in zip(bboxes, labels, ids):
        cx, cy, w, h = bbox
        x0, y0 = (cx - w / 2) * image.width, (cy - h / 2) * image.height
        x1, y1 = (cx + w / 2) * image.width, (cy + h / 2) * image.height

        # Calcular el color para la categoría
        color = COLORS[label]

        # Dibujar el bounding box
        rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=2)
        ax.add_patch(rect)

        # Mostrar ID del objeto y nombre de la categoría
        ax.text(x0, y0 - 10, f"ID: {obj_id}, {COCO_INSTANCE_CATEGORY_NAMES[label]}", fontsize=12, color='white',
                bbox=dict(facecolor=color, alpha=0.7))

    ax.axis('off')  # Ocultar ejes
    return fig
