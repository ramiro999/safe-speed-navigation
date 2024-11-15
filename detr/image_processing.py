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

def plot_detr_results(image, bboxes, labels):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)

    for bbox, label in zip(bboxes, labels):
        cx, cy, w, h = bbox
        x0, y0 = (cx - w / 2) * image.width, (cy - h / 2) * image.height
        x1, y1 = (cx + w / 2) * image.width, (cy + h / 2) * image.height

        # Calcular el área en píxeles asegurando valores positivos y redondeando
        pixel_area = abs(round((x1 - x0) * (y1 - y0)))

        ax.text(x0, y1 + 10, f'Area:{pixel_area}px²', fontsize=10, color='black', bbox=dict(facecolor='white', alpha=0.7))


        color = COLORS[label]
        rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x0, y0, COCO_INSTANCE_CATEGORY_NAMES[label], fontsize=14, color='black',
                bbox=dict(facecolor=color, alpha=0.7))

    ax.axis('off')  # Ocultar ejes
    #fig.tight_layout(pad=0)  # Ajustar el gráfico sin relleno
    return fig
