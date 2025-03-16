import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor
import torchvision.transforms as T
from torchvision.transforms import ToTensor
import cv2
import time  # Para medir tiempos y verificar si se está congelando
torch.set_num_threads(12)  # Usa 12 de los 14 núcleos para entrenamiento
from tqdm import tqdm
import torch.backends.mkldnn

torch.backends.mkldnn.enabled = True

# Dataset KITTI para carga de datos
dataset_path = '../dataset'

CLASS_MAPPING = {
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

def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    max_height = max(img.shape[1] for img in images)
    max_width = max(img.shape[2] for img in images)

    padded_images = [
        T.Pad((0, 0, max_width - img.shape[2], max_height - img.shape[1]))(img)
        for img in images
    ]

    batch_images = torch.stack(padded_images)
    batch_targets = []

    for target in targets:
        batch_targets.append({
            'boxes': target['boxes'],
            'class_labels': target['class_labels']
        })
    
    return batch_images, batch_targets

class KITTIDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.image_filenames = sorted(os.listdir(images_dir))
        self.label_filenames = sorted(os.listdir(labels_dir))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_filenames[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = ToTensor()(image)

        label_path = os.path.join(self.labels_dir, self.label_filenames[idx])
        boxes = []
        labels = []
        with open(label_path, 'r') as file:
            for line in file.readlines():
                parts = line.strip().split(' ')
                label = parts[0]  # Tipo (e.g., Pedestrian)
                if label not in CLASS_MAPPING:
                    continue
                x_min, y_min, x_max, y_max = map(float, parts[4:8])
                width = x_max - x_min
                height = y_max - y_min
                labels.append(CLASS_MAPPING[label])
                boxes.append([x_min, y_min, width, height])

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'class_labels': torch.tensor(labels, dtype=torch.int64)  # Cambio clave
        }

        return image, target

# Configuración del modelo y procesador
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RTDetrV2ForObjectDetection.from_pretrained('PekingU/rtdetr_v2_r50vd').to(device)
processor = RTDetrImageProcessor.from_pretrained('PekingU/rtdetr_v2_r50vd')

# Dataloader para carga de datos
dataset = KITTIDataset(images_dir='../dataset/images', labels_dir='../dataset/labels')
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn, num_workers=8, pin_memory=True)

# Verificar carga de datos
#print("Verificando la carga de datos...")
#for batch_idx, (images, targets) in enumerate(dataloader):
#    print(f"✔️ Batch {batch_idx + 1} cargado correctamente")
#    if batch_idx == 2:  # Probar solo los primeros 2 batches
#        break

def train(model, dataloader, epochs=10):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    for epoch in range(epochs):
        print(f"Iniciando Epoch {epoch + 1}/{epochs}")

        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1}")

        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)  # Optimiza transferencia en CPU
            for t in targets:
                t['boxes'] = t['boxes'].to(device, non_blocking=True)
                t['class_labels'] = t['class_labels'].to(device, non_blocking=True)

            pixel_values = processor(images, return_tensors='pt', do_rescale=False).pixel_values.to(device)

            loss_dict = model(pixel_values=pixel_values, labels=targets)
            loss = loss_dict.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"     Batch {batch_idx + 1} - Loss: {loss.item()}")

#model.gradient_checkpointing_enable(False)
torch.backends.mkldnn.enabled = True


def evaluate(model, dataloader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, targets in dataloader:
            pixel_values = processor(images, return_tensors='pt', do_rescale=False).pixel_values.to(device)
            loss_dict = model(pixel_values=pixel_values, labels=targets)
            total_loss += loss_dict.loss.item()
    print(f'Validation Loss: {total_loss / len(dataloader)}')

# Ejecución
torch.manual_seed(42)
train(model, dataloader)
evaluate(model, dataloader)
