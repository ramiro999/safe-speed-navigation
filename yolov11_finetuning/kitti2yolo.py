import os

# Definir las clases usadas en YOLO
CLASSES = ["Car", "Pedestrian", "Cyclist", "Van", "Truck", "Misc", "Tram", "Person_sitting", "DontCare"]

def convert_kitti_to_yolo(kitti_label_path, yolo_label_path, image_width=1242, image_height=375):
    with open(kitti_label_path, "r") as file:
        lines = file.readlines()
    
    with open(yolo_label_path, "w") as file:
        for line in lines:
            values = line.split()
            class_name = values[0]

            # Ignorar clases que no sean Car, Pedestrian, Cyclist
            if class_name not in CLASSES:
                continue

            class_id = CLASSES.index(class_name)
            x1, y1, x2, y2 = map(float, values[4:8])

            # Convertir de formato KITTI (coordenadas absolutas) a YOLO (normalizado)
            x_center = ((x1 + x2) / 2) / image_width
            y_center = ((y1 + y2) / 2) / image_height
            width = (x2 - x1) / image_width
            height = (y2 - y1) / image_height

            file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

# Convertir todas las etiquetas en la carpeta de entrenamiento
kitti_labels_dir = "../sample_data/train_labels_kitti"
yolo_labels_dir = "../sample_data/train_labels_yolo"

os.makedirs(yolo_labels_dir, exist_ok=True)

for file_name in os.listdir(kitti_labels_dir):
    if file_name.endswith(".txt"):
        kitti_path = os.path.join(kitti_labels_dir, file_name)
        yolo_path = os.path.join(yolo_labels_dir, file_name)
        convert_kitti_to_yolo(kitti_path, yolo_path)

print("✅ Conversión completada. Ahora usa los nuevos labels en YOLO.")
