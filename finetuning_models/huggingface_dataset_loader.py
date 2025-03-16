from datasets import Dataset, Features, Image, Value, Sequence
import pandas as pd
import ast
import os
import matplotlib.pyplot as plt
import PIL.Image as PILImage

# Ruta local a tu dataset
DATASET_PATH = "../finetuning_models"

# Cargar el dataset desde el CSV
data = pd.read_csv(os.path.join(DATASET_PATH, 'dataset.csv'))

# Convertir la columna 'objects' a diccionarios reales
data['objects'] = data['objects'].apply(ast.literal_eval)

# Actualizar la columna 'image' con las rutas de imágenes directamente
data['image'] = data['image'].apply(lambda x: x.replace('../dataset/', DATASET_PATH + '/'))

# Definir el esquema del dataset
features = Features({
    "image_id": Value("string"),
    "image": Image(),  # Define la columna 'image' como imagen directamente
    "width": Value("int32"),
    "height": Value("int32"),
    "objects": {
        "id": Sequence(Value("int32")),
        "area": Sequence(Value("float32")),
        "bbox": Sequence(Sequence(Value("float32"))),
        "category": Sequence(Value("int32")),
    }
})

# Convertir el DataFrame en Dataset de Hugging Face
hf_dataset = Dataset.from_pandas(data, features=features)

sample_image = PILImage.open(data['image'].iloc[0])
#plt.imshow(sample_image)
#plt.axis('off')
#plt.show()

# Iniciar sesión en Hugging Face
from huggingface_hub import login
login()  # Introduce tu token de Hugging Face aquí cuando te lo solicite

# Subir el dataset a Hugging Face
hf_dataset.push_to_hub("KingRam/Kitti-Object-Detection-Evaluation-2012")
