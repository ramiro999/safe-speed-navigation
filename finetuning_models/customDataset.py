import os
import csv
from PIL import Image
import pandas as pd

# Mapeo de categorías según tu descripción
CATEGORY_MAP = {
    "Car": 0,
    "Pedestrian": 1,
    "Cyclist": 2,
    "Van": 3,
    "Truck": 4,
    "Misc": 5,
    "Tram": 6,
    "Person_sitting": 7,
    "DontCare": 8
}

# Directorios del dataset
DATASET_DIR = "../dataset"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")

# Lista para almacenar los datos
dataset = []

# Contador global para ids de anotaciones
annotation_id = 0

def parse_label_file(label_path):
    global annotation_id
    ids, areas, bboxes, categories = [], [], [], []
    with open(label_path, 'r') as file:
        for line in file.readlines():
            parts = line.strip().split()
            category = parts[0]
            if category not in CATEGORY_MAP:
                continue
            x_min, y_min, x_max, y_max = map(float, parts[4:8])
            width = x_max - x_min
            height = y_max - y_min
            area = width * height
            ids.append(annotation_id)
            areas.append(area)
            bboxes.append([x_min, y_min, width, height])
            categories.append(CATEGORY_MAP[category])
            annotation_id += 1
    return {"id": ids, "area": areas, "bbox": bboxes, "category": categories}

# Procesar las imágenes y labels
for image_file in sorted(os.listdir(IMAGES_DIR)):
    if image_file.endswith(".png"):
        image_id = image_file
        image_path = os.path.join(IMAGES_DIR, image_file)
        label_path = os.path.join(LABELS_DIR, image_file.replace(".png", ".txt"))

        with Image.open(image_path) as img:
            width, height = img.size

        objects = parse_label_file(label_path) if os.path.exists(label_path) else {"id": [], "area": [], "bbox": [], "category": []}

        dataset.append({
            "image_id": image_id,
            "image": image_path,
            "width": width,
            "height": height,
            "objects": objects
        })

# Crear el DataFrame y exportar a CSV
df = pd.DataFrame(dataset)
df.to_csv("dataset.csv", index=False)

print("Dataset generado con éxito: 'dataset.csv'")
