from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split

img_path = Path("../sample_data/images")
label_path = Path("../sample_data/train_labels_yolo")

train_path = Path("../sample_data/train")
valid_path = Path("../sample_data/valid")

train_path.mkdir(exist_ok=True)
valid_path.mkdir(exist_ok=True)

ims = sorted(list(img_path.glob('*')))
labels = sorted(list(label_path.glob('*')))
pairs = list(zip(ims, labels))

# Dividir en 80% train y 20% valid
train, valid = train_test_split(pairs, test_size=0.2, shuffle=True)

# Mover archivos a carpetas correctas
for t_img, t_lb in train:
    shutil.copy(t_img, train_path / t_img.name)
    shutil.copy(t_lb, train_path / t_lb.name)

for v_img, v_lb in valid:
    shutil.copy(v_img, valid_path / v_img.name)
    shutil.copy(v_lb, valid_path / v_lb.name)

print("✅ Datos organizados correctamente.")

### ----------------- Crear archivo YAML ----------------- ###

yaml_file = """names:
- Car
- Pedestrian
- Cyclist
- Van
- Truck
- Misc
- Tram
- Person_sitting
- DontCare
nc: 9
train: ../sample_data/train
val: ../sample_data/valid
"""

with open("../sample_data/kitti.yaml", "w") as f:
    f.write(yaml_file)

print("✅ kitti.yaml creado correctamente.")
