#app.py

# Importar librerías necesarias
import sys
import os
import torch
import gradio as gr
import cv2
import numpy as np
import tempfile
import plotly.express as px
import plotly.graph_objects as go
from typing import Union, Iterable
from gradio.themes.utils import colors, fonts, sizes
from gradio.themes.base import Base
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from glob import glob
import shutil
from gradio_modal import Modal
import random

# Añadir directorios al path
sys.path.extend([
    os.path.join(os.path.dirname(__file__), 'stereo_estimation', 'NMRF'),
    os.path.join(os.path.dirname(__file__), 'stereo_estimation', 'NMRF', 'ops', 'setup', 'MultiScaleDeformableAttention')
])

# Importar módulos personalizados
from safety_calculator import calculate_lookahead_distance
from rt_detrv2.model_rt_detrv2 import load_rtdetrv2_model, load_image_processor
from yolov11_finetuning.model_yolo_f import load_yolov11_model_f
from yolov11.model_yolo import load_yolov11_model
from stereo_estimation.NMRF.disparity_inference import run_inference

"""
Función main para la interfaz de Gradio.
"""


# Cargar el modelo RT-DETRv2
rt_detrv2 = load_rtdetrv2_model()
rt_detrv2_image_processor = load_image_processor()

# Cargar el modelo YOLOv11 Fine-Tuning
yolov11_model_kitti = load_yolov11_model_f() 

# Cargar el modelo YOLOv11
yolov11_model = load_yolov11_model()

# Variable para el modelo seleccionado
selected_model = "RT-DETRv2" # Esto se actualizará según la selección del usuario

# Variable global para almacenar la imagen de salida de Stereo Inference
stereo_output_image = None
stereo_output_pred = None
image_path_left_original = None
objects_info = []  # Almacenar los objetos detectados en la imagen

# Datos del dataset KITTI
KITTI_FOCAL_LENGTH = 725.0087
KITTI_BASELINE = 0.532725
KTTI_IMAGES_LEFT = [
"./images/stereo_images/kitti_images_left/000004_10.png",
"./images/stereo_images/kitti_images_left/0000000000.png",
"./images/stereo_images/kitti_images_left/0000000033.png",
"./images/stereo_images/kitti_images_left/0000000047.png",
"./images/stereo_images/kitti_images_left/000004_10.png",
"./images/stereo_images/kitti_images_left/000004_10.png",
]
KITTI_IMAGES_RIGHT = [
"./images/stereo_images/kitti_images_right/000004_10.png",
"./images/stereo_images/kitti_images_right/0000000000.png",
"./images/stereo_images/kitti_images_right/0000000033.png",
"./images/stereo_images/kitti_images_right/0000000047.png",
"./images/stereo_images/kitti_images_right/000004_10.png",
"./images/stereo_images/kitti_images_right/000004_10.png",
]

# Datos del dataset DrivingStereo
DRIVING_STEREO_FOCAL_LENGTH = 2007.113
DRIVING_STEREO_BASELINE = 0.54
DRIVING_STEREO_IMAGES_LEFT = [
"./images/stereo_images/driving_stereo_images_left/2018-08-17-09-45-58_2018-08-17-10-22-09-789.png",
"./images/stereo_images/driving_stereo_images_left/2018-10-19-09-30-39_2018-10-19-09-38-29-439.png",
"./images/stereo_images/driving_stereo_images_left/2018-10-25-07-37-26_2018-10-25-07-51-07-850.png",
"./images/stereo_images/driving_stereo_images_left/2018-10-31-06-55-01_2018-10-31-06-59-29-671.png",
"./images/stereo_images/driving_stereo_images_left/2018-07-11-14-48-52_2018-07-11-14-50-33-885.png",
]
DRIVING_STEREO_IMAGES_RIGHT = [
"./images/stereo_images/driving_stereo_images_right/2018-08-17-09-45-58_2018-08-17-10-22-09-789.png",
"./images/stereo_images/driving_stereo_images_right/2018-10-19-09-30-39_2018-10-19-09-38-29-439.png",
"./images/stereo_images/driving_stereo_images_right/2018-10-25-07-37-26_2018-10-25-07-51-07-850.png",
"./images/stereo_images/driving_stereo_images_right/2018-10-31-06-55-01_2018-10-31-06-59-29-671.png",
"./images/stereo_images/driving_stereo_images_right/2018-07-11-14-48-52_2018-07-11-14-50-33-885.png",
]

object_distance = None # Distancia del objeto al sistema óptico en metros
object_distance_mm = None  # Distancia del objeto al sistema óptico en milímetros
object_height = None # Altura del objeto en pixeles

# ---- KITTI ---

pixSize = None  # Tamaño del pixel en milimetros
focalLengthPixels = None # Longitud focal en pixeles
focalLength = None  # Longitud focal en milimetros
imageHeight = None  # Altura de la imagen en pixeles

sensorSize = None # Tamaño del sensor en milimetros

gsd = None  # Ground Sampling Distance (GSD) en centrimetros por pixel
hp = None  # Altura del objeto en milimetros para eliminar la dependencia de mm/pixel
hp = None  # Altura del objeto en metros

dataset_selection = gr.Radio(["KITTI", "Driving Stereo"], label="Select Dataset", value="KITTI")

def save_temp_image(image_array, output_name=None, start_number=5):
    """
    Guarda una imagen numpy.ndarray en un archivo temporal y devuelve la ruta del archivo.
    """
    temp_dir = tempfile.gettempdir()

    if not hasattr(save_temp_image, "counter"):
        # Buscar el número más alto en la carpeta temporal
        existing_files = [f for f in os.listdir(temp_dir) if f.endswith(".png") and f.isdigit()]
        existing_numbers = [int(f) for f in existing_files] if existing_files else []
        save_temp_image.counter = max(existing_numbers, default=start_number - 1) + 1  # Continuar numeración

    if output_name is None:
        output_name = f"{save_temp_image.counter:010d}.png"

    temp_file_path = os.path.join(temp_dir, output_name)
    cv2.imwrite(temp_file_path, image_array)

    save_temp_image.counter += 1  # Aumentar el contador correctamente
    return temp_file_path

def stereo_inference(image_path_left=None, image_path_right=None):
    """
    Realiza inferencia estéreo. Soporta tanto rutas de archivo como imágenes cargadas en memoria (numpy.ndarray).
    """
    # Comprobar si las entradas son numpy.ndarray y se guardan archivos temporales si es necesario.
    if isinstance(image_path_left, np.ndarray):
        image_path_left = save_temp_image(image_path_left)
    if isinstance(image_path_right, np.ndarray):
        image_path_right = save_temp_image(image_path_right)

    # Comprobar de que las imagenes se carguen correctamente
    if not isinstance(image_path_left, str) or not isinstance(image_path_right, str):
        return gr.Warning("Inputs must be file paths or numpy.ndarray images.")

    global stereo_output_image, image_path_left_original, stereo_output_pred # Variables globales para almacenar la imagen de salida, la imagen original y el valor de disparidad
    image_path_left_original = image_path_left 
    dataset_name = "kitti" 
    output = "./outputs/disparity_results" # Carpeta de salida para los resultados
    resume_path = "./stereo_estimation/NMRF/pretrained/kitti.pth" # Ruta del modelo pre-entrenado con kitti.pth

    # Crear una lista con las imágenes de entrada proporcionadas por el usuario
    image_list = [(image_path_left, image_path_right)]

    # Realizar la inferencia usando las imágenes proporcionadas
    disp_preds = run_inference(dataset_name, output, resume_path, image_list, show_attr="disparity")

    # Buscar la última imagen generada en la carpeta de resultados
    result_files = [os.path.join(output, f) for f in os.listdir(output) if f.endswith(".png")]
    if result_files:
        # Ordenar por fecha de modificación
        result_files.sort(key=os.path.getmtime, reverse=True)
        stereo_output_image = os.path.abspath(result_files[0])

        stereo_output_pred = disp_preds[0]  # Guardar la matriz de disparidad en la variable global

        #print(f"Disparity prediction saved at: {disp_pred_path}")

        print(f"Rangos minimo y maximo del disp_pred: {disp_preds[0].min()}, {disp_preds[0].max()}")

        # Mostrar la imagen de disparidad con un colorbar
        disparity_image = cv2.imread(stereo_output_image, cv2.IMREAD_GRAYSCALE)
        plt.figure(figsize=(10, 5))
        plt.imshow(disparity_image)
        plt.colorbar(label='Disparity (pixels)')
        plt.axis('off')
        plt.close()

        return gr.update(value=stereo_output_image, visible=True)

    # En caso de que no se encuentre un archivo, lanzar un error
    raise FileNotFoundError("No generated image was found in the results folder.")

def get_camera_parameters(selected_dataset, custom_focal_length =None, custom_baseline=None):

    """
    Obtiene los parámetros de la cámara (focal length y baseline).
    """

    if custom_focal_length is not None and custom_baseline is not None:
        try:
            return float(custom_focal_length), float(custom_baseline)
        except ValueError:
            return None, None

    # Si no se ingresan valores manuales, usar los valores por defecto del dataset
    if selected_dataset == "KITTI":
        return KITTI_FOCAL_LENGTH, KITTI_BASELINE
    elif selected_dataset == "Driving Stereo":
        return DRIVING_STEREO_FOCAL_LENGTH, DRIVING_STEREO_BASELINE
    elif selected_dataset == "Own images":
        return None, None  
    else:
        return None, None


def generate_depth_map(disparity_path=None, focal_length=None, baseline=None):
    """
    Genera un mapa de profundidad con una barra de colores en el rango de metros 0-100.
    """
    global stereo_output_pred

    if stereo_output_pred is None:
        raise ValueError("No disparity data available.")
    
    if focal_length is None or baseline is None:
        return gr.Warning("⚠️ Please select a dataset or provide your own camera parameters.")

    #print(f"✅ Parámetros recibidos en `generate_depth_map`: Focal Length = {focal_length}, Baseline = {baseline}")

    # Usar la matriz de disparidad global si no se proporciona una ruta
    disparity = stereo_output_pred

    # Evitar divisiones por cero y valores muy pequeños
    min_disparity = 1e-8
    disparity[disparity < min_disparity] = min_disparity

    # Calcular el mapa de profundidad
    depth_map = (focal_length * baseline) / disparity

    depth_map = np.round(depth_map.numpy() * 256).astype(np.uint16)
    depth_map = depth_map.astype(np.float32) / 256.0

    print(f"Valor mínimo y máximo del depth map en `generate_depth_map`: {depth_map.min()}, {depth_map.max()}")

    # Crear figura con barra de colores
    fig, ax = plt.subplots(dpi=300, frameon=False)
    im = ax.imshow(depth_map, cmap='inferno_r')
    
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', label='Depth (meters)', pad=0.02)  # Ajustar el pad para acercar la barra de colores
    cbar.ax.xaxis.label.set_color('white')
    cbar.ax.tick_params(color='white')  
    cbar.ax.xaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color='white')  # Cambiar el color de las etiquetas a blanco

    # Cambiar el color de las etiquetas a negro si el tema del PC es claro
    if plt.rcParams['axes.facecolor'] == 'white':
        cbar.ax.xaxis.label.set_color('black')
        cbar.ax.tick_params(color='black')  
        cbar.ax.xaxis.set_tick_params(color='black')
        plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color='black')  # Cambiar el color de las etiquetas a negro

    # Hacer la letra más pequeña
    cbar.ax.tick_params(labelsize=6)
    cbar.ax.xaxis.label.set_size(6)

    ax.axis('off')

    # Eliminar completamente márgenes blancos
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)

    # Ajustar diseño para eliminar cualquier espacio
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0.1)

    # Guardar el resultado
    output_path_with_colorbar = "./outputs/depth_with_colorbar.png"
    plt.savefig(output_path_with_colorbar, bbox_inches='tight', pad_inches=0, dpi=300, transparent=True)
    plt.close()

    #print(f"✅ Mapa de profundidad guardado en: {output_path_with_colorbar}")

    return output_path_with_colorbar


def only_depth_map(disparity_path=None, focal_length=None, baseline=None, input_image_path=None):
    """
    Genera un mapa de profundidad a partir de un mapa de disparidad.

    Args:
        disparity_path (str, opcional): Ruta al archivo de disparidad (.npy).
        focal_length (float): Longitud focal de la cámara.
        baseline (float): Línea base (distancia entre las cámaras).
        input_image_path (str, opcional): Ruta de la imagen de entrada para usar su nombre.

    Returns:
        tuple: (depth_map, output_path) donde depth_map es el mapa de profundidad y output_path es la ruta del archivo guardado.
    """
    global stereo_output_pred, stereo_output_image

    if stereo_output_pred is None or stereo_output_image is None:
        raise ValueError("No disparity data available.")

    if focal_length is None or baseline is None:
        print("❗️ Warning: Focal length or baseline not provided.")
        return None, None  # Garantiza que siempre se retornen dos valores.

    # Usar la matriz de disparidad global si no se proporciona una ruta
    disparity = stereo_output_pred

    #print(f"Valores mínimos y máximos en la función only_depth_map: {disparity.min()}, {disparity.max()}")

    # Evitar divisiones por cero y valores muy pequeños
    min_disparity = 1e-8
    disparity[disparity < min_disparity] = min_disparity

    # Calcular el mapa de profundidad
    depth_map = (focal_length * baseline) / disparity

    #print(f"Valores mínimos y máximos del depth map en la función only_depth_map: {depth_map.min()}, {depth_map.max()}")

    base_name = os.path.splitext(os.path.basename(stereo_output_image))[0]
    output_name = f"{base_name}.png"

    # Ruta de salida
    output_dir = "./outputs/depth_results"
    output_path = os.path.join(output_dir, output_name)

    # Guardar el mapa de profundidad como imagen en uint16
    save_depth_map = np.round(depth_map.numpy() * 256).astype(np.uint16)
    Image.fromarray(save_depth_map, mode='I;16').save(output_path)

    #print(f"✅ Mapa de profundidad guardado en: {output_path}")

    return depth_map, output_path

def clear_folder(folder_path):
    """
    Borra todos los archivos dentro de una carpeta.
    """
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Borra archivos y enlaces simbólicos
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Borra carpetas completas
            except Exception as e:
                print(f"❌ Error al borrar {file_path}: {e}")

# Función para detección de objetos y calcular la distancia usando la disparidad
def object_detection_with_disparity(selected_model_name, selected_dataset, focal_length, baseline):
    global stereo_output_image, image_path_left_original, objects_info
    if not stereo_output_image:
        return None, gr.Warning("A Stereo Inference output image has not been generated.")
    
    focal_length, baseline = get_camera_parameters(selected_dataset, focal_length, baseline)

    # Generar el mapa de profundidad Z (en metros)
    depth, depth_map_path = only_depth_map(stereo_output_image, focal_length, baseline)

    # Verificar que el depth_map se generó correctamente
    if depth is None or depth_map_path is None:
        return None, gr.Warning("⚠️ Depth map generation failed. Please check the provided parameters.")

    image = cv2.imread(image_path_left_original, cv2.IMREAD_COLOR)

    # Definición de clases
    class_rt_detrv2 = list(rt_detrv2.config.id2label.values())
    class_kitti = ['Pedestrian', 'Cyclist', 'Car', 'Person_sitting', 'Van', 'Misc', 'Tram', 'Truck', 'DontCare']

    # Selección del modelo YOLO según el dataset
    if selected_dataset == "KITTI":
        selected_yolo_model = yolov11_model_kitti
        class_yolo = class_kitti
        #print(f"Selected KITTI Classes: {class_yolo}")
    else:
        selected_yolo_model = yolov11_model
        class_yolo = list(yolov11_model.model.names.values())
        #print(f"Selected Default YOLOv11 Classes: {class_yolo}")

    # Selección de las clases según el modelo
    if selected_model_name == "RT-DETRv2":
        class_names = class_rt_detrv2
    elif selected_model_name == "YOLOv11":
        class_names = class_kitti if selected_dataset == "KITTI" else list(yolov11_model.model.names.values())
    else:
        return None, gr.Warning("Modelo no válido seleccionado.")

    bboxes, labels = [], []  # Inicializar variables comunes

    if selected_model_name == "RT-DETRv2":
        # Procesar con RT-DETRv2
        image_tensor = rt_detrv2_image_processor(images=image, return_tensors='pt')
        with torch.no_grad():
            outputs = rt_detrv2(**image_tensor)
        #print(outputs.keys())
        probas = outputs['logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.8
        bboxes = outputs['pred_boxes'][0, keep].cpu().numpy()
        labels = probas[keep].argmax(-1).cpu().numpy()

    elif selected_model_name == "YOLOv11":
        # Procesar con YOLOv11
        results = yolov11_model(image_path_left_original)
        bboxes = [box.xyxy[0].tolist() for box in results[0].boxes]
        labels = [int(box.cls[0]) for box in results[0].boxes]
    else:
        return None, gr.Warning("Invalid model selected.")

    # Procesar información de objetos detectados
    objects_info = []

    for idx, (bbox, label) in enumerate(zip(bboxes, labels), start=1):
        if selected_dataset == "KITTI" and (label < 0 or label >= len(class_kitti)):
            #print(f"Objeto desconocido detectado con etiqueta {label}. Se omitirá este objeto.")
            continue 

        if selected_model_name == "RT-DETRv2":
            if label < 0 or label >= len(class_names):
                #print(f"Invalid label detected: {label}")
                continue
            cx, cy, w, h = bbox
            x0, y0 = int((cx - w / 2) * image.shape[1]), int((cy - h / 2) * image.shape[0])
            x1, y1 = int((cx + w / 2) * image.shape[1]), int((cy + h / 2) * image.shape[0])
        elif selected_model_name == "YOLOv11":
            x0, y0, x1, y1 = map(int, bbox)

        height_bb = abs(y1 - y0) # Poner cuanto vale en metros la altura de la caja delimitadora

        # Recortar región del mapa de profundidad correspondiente al bounding box
        bbox_depth = depth[y0:y1, x0:x1]
        bbox_valid = bbox_depth[bbox_depth > 0]

        # Calcular distancia media en región válida
        if len(bbox_valid) > 0:
            median_distance = np.median(bbox_valid)
        else:
            median_distance = float('inf')

        # Agregar información del objeto detectado oerdenado 
        objects_info.append({
            'id': idx,
            'class': class_names[label],
            'height': height_bb,
            'bbox': [x0, y0, x1, y1],
            'distance': median_distance
        })
        #print(f"Label value: {label}, Max index in class_names: {len(class_names) - 1}")

    # Ordenar los objetos por distancia y reasignar IDs
    objects_info.sort(key=lambda x: x['distance'])
    for new_idx, obj in enumerate(objects_info, start=1):
        obj['id'] = new_idx

    # Construir tarjetas HTML
    cards_html = """
    <style>
        .grid-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            padding: 20px;
        }
        .card {
            background-color: #2c2f33;
            color: #ffffff;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            text-align: left;
        }

        .card h3 {
            margin: 10px 0;
            font-size: 18px;
            display: flex;
            align-items: center;
            color: inherit;
        }
        .card h3 img {
            width: 24px;
            margin-right: 10px;
        }
        .card p {
            font-size: 16px;
            margin: 5px 0;
            color: inherit;
        }

        @media (prefers-color-scheme: light) {
        .card {
            background-color: #f8f9fa;
            color: #000000;
        }
        .card h3 {
            color: #333333;
        }
        .card p {
            color: #555555;
            }

        .card:hover {
            transform: scale(1.05);
            opacity: 0.9;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3); /* Sombra más pronunciada */
            }
        }
    </style>
    <div class="grid-container">
    """

    CLASS_IMAGES = {
        'car': 'https://img.icons8.com/color/48/000000/car--v1.png',
        'traffic light': 'https://i.ibb.co/KG6PrDH/icons8-traffic-lights-sign-96.png',
        'person': 'https://i.ibb.co/3zFHGg2/icons8-person-96-2.png',
        'bicycle': 'https://i.ibb.co/d4tnFHP/icons8-bicycle-100.png',
        'motorcycle':'https://i.ibb.co/MxCf7LCT/icons8-motorbike-96.png',
        'motorbike': 'https://i.ibb.co/MxCf7LCT/icons8-motorbike-96.png',
        'bus': 'https://i.ibb.co/0f20RxR/icons8-bus-96.png',
        'truck': 'https://i.ibb.co/HD53Wtq/icons8-truck-96.png',
        'Car': 'https://img.icons8.com/color/48/000000/car--v1.png',
        'Pedestrian': 'https://i.ibb.co/3zFHGg2/icons8-person-96-2.png',
        'Cyclist': 'https://i.ibb.co/d4tnFHP/icons8-bicycle-100.png',
        'Van': 'https://i.ibb.co/67zBH74g/icons8-van-96.png',
        'Truck': 'https://i.ibb.co/HD53Wtq/icons8-truck-96.png',
        'Misc': 'https://i.ibb.co/G4n4jMfP/icons8-m-64.png',
        'Tram': 'https://i.ibb.co/xKz6KXkT/icons8-tram-96.png',
        'Person_sitting': 'https://i.ibb.co/qY9ygdD6/icons8-sitting-on-chair-96.png',
        'DontCare': 'https://i.ibb.co/5XFY7GF7/icons8-see-no-evil-monkey-96.png'
    }

    def get_class_icon(class_name):
        return CLASS_IMAGES.get(class_name, 'https://i.ibb.co/5XFY7GF7/icons8-see-no-evil-monkey-96.png')
    
    if  selected_dataset == "KITTI":
        pixSize = 4.65e-3
        focalLengthPixels = 725.0087
        focalLength = focalLengthPixels * pixSize
        imageHeight = 375
        sensorSize = pixSize * imageHeight
    elif selected_dataset == "Driving Stereo":
        pixSize = 5.86e-3
        focalLengthPixels = 2007.113
        focalLength = focalLengthPixels * pixSize
        imageHeight = 800
        sensorSize = pixSize * imageHeight
    else:
        pixSize, focalLength, imageHeight, sensorSize = None, None, None, None

    if focalLength and imageHeight:
        for obj in objects_info:
            object_distance_mm = obj['distance'] * 1000
            gsd = (object_distance_mm * sensorSize) / (focalLength * imageHeight)
            object_height_meters = obj['height'] * gsd / 1000

            obj['height_meters'] = object_height_meters

            cards_html += f"""
            <div class="card">
                <h3>
                    <img src={get_class_icon(obj['class'])} alt="Object Icon">
                    Object ID: {obj['id']}
                </h3>
                <p><strong>Class:</strong> {obj['class']}</p>
                <p><strong>Height:</strong> {obj['height']} pixels - {object_height_meters:.2f} m</p>
                <p><strong>Coordinates:</strong> ({obj['bbox'][0]}, {obj['bbox'][1]}) to ({obj['bbox'][2]}, {obj['bbox'][3]})</p>
                <p><strong>Distance:</strong> {obj['distance']:.2f} meters</p>
            </div>
            """

    cards_html += "</div>"

    # Visualizar y guardar resultados en Plotly
    fig = go.Figure()

    # Añadir la imagen original al gráfico
    fig.add_trace(go.Image(z=image))

    # Añadir las cajas delimitadoras sobre la imagen
    for obj in objects_info:
        x0, y0, x1, y1 = obj['bbox']
        color = px.colors.qualitative.Plotly[class_names.index(obj['class']) % len(px.colors.qualitative.Plotly)]
        fig.add_shape(
            type="rect",
            x0=x0, y0=y0, x1=x1, y1=y1,
            line=dict(color=color, width=2),
            fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)"  # Color con transparencia
        )

        # Añadir el texto con la distancia, ID y clase sobre la caja
        fig.add_annotation(
            x=(x0 + x1) / 2, y=y0 - 10,
            text=f"ID: {obj['id']} Class: {obj['class']} Dist: {obj['distance']:.2f} m",
            showarrow=False,
            font=dict(color=color, size=12),
            bgcolor="rgba(0, 0, 0, 0.5)",
            bordercolor=color,
            borderwidth=1,
            borderpad=2
        )

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),  # Eliminar márgenes
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )

    # Guardar el gráfico como imagen
    fig.write_image("./outputs/object_detection/object_detection_results_with_distances.png")

    return fig, cards_html

# Función para el cálculo de distancia segura y gráficos de decisión
def calculate_distance(mu, t, l, B, turning_car, cog, wheelbase, selected_object_id):
    global objects_info

    if not selected_object_id:
        return gr.Warning("Please select at least one detected object to calculate the distance."), None, None, None

    # Asegurarse de que el ID seleccionado existe en objects_info
    selected_objects = [obj for obj in objects_info if obj['id'] == selected_object_id]

    if not selected_objects:
        return gr.Warning("⚠️ Selected object ID not found in the detected objects."), None, None, None

    # Obtener altura y distancia del objeto seleccionado
    obj = selected_objects[0]
    object_height = obj.get('height_meters', None)
    object_distance = obj['distance']

    if object_distance is None or object_distance == float('inf'):
        return gr.Warning("⚠️ Invalid object distance, please select another object."), None, None, None

    # Calcular las distancias de seguridad
    try:
        fig1, fig2, fig3, fig4, selected_ifov, safe_speed_stop, safe_speed_swerve, safe_speed_stop_haov, safe_speed_swerve_haov, safe_speed_stop_vaov, safe_speed_swerve_vaov, object_height_meters = calculate_lookahead_distance(
            mu=mu, t=t, l=l, B=B, cog=cog, wheelbase=wheelbase, 
            turning_angle=turning_car, object_height=object_height, 
            object_distance=object_distance, image_path=None
        )
        return fig1, fig2, fig3, fig4, selected_ifov, safe_speed_stop, safe_speed_swerve, safe_speed_stop_haov, safe_speed_swerve_haov, safe_speed_stop_vaov, safe_speed_swerve_vaov, object_height_meters
    except Exception as e:
        return gr.Warning(f"Error during calculation: {str(e)}"), None, None, None


# Función para actualizar los parámetros del vehículo según el modelo seleccionado
def update_vehicle_params(vehicle_model):
    vehicle_params = {
        "Volkswagen Passat (B6)": (11.4, 0.55, 2.71),
        "Honda Aviancer": (11.6, 0.66, 2.82),
        "Tesla S": (11.8, 0.46, 2.96),
        "Toyota Supra": (10.40, 0.4953, 2.47),
    }
    return vehicle_params.get(vehicle_model, (11.4, 0.55, 2.71))  # Por defecto Volkswagen Passat (B6)

# Diseño de la interfaz de Gradio
class Seafoam(Base):
    def __init__(
        self,
        *,
        primary_hue: Union[colors.Color, str] = colors.emerald,
        secondary_hue: Union[colors.Color, str] = colors.blue,
        neutral_hue: Union[colors.Color, str] = colors.gray,
        spacing_size: Union[sizes.Size, str] = sizes.spacing_md,
        radius_size: Union[sizes.Size, str] = sizes.radius_md,
        text_size: Union[sizes.Size, str] = sizes.text_lg,
        font: Union[fonts.Font, str, Iterable[Union[fonts.Font, str]]] = (
            fonts.GoogleFont("Quicksand"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono: Union[fonts.Font, str, Iterable[Union[fonts.Font, str]]] = (
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )

seafoam = Seafoam()

custom_css = """
<style>
    .gradio-tabs {
        display: flex;
        justify-content: center; /* Centra los tabs horizontalmente */
    }
</style>
"""

with gr.Blocks(theme=seafoam, css=custom_css) as demo:
    
    with Modal(visible=True) as modal:
        gr.HTML("""
        <div style="
            animation: slideUpFadeIn 0.6s ease-out;
            background-color: #ffffff;
            color:rgb(102, 131, 159);
            padding: 30px;
            border-radius: 20px;
            box-shadow: 0 12px 35px rgba(17, 50, 199, 0.3);
            text-align: left;
            border-left: 6px solid #0e3a67;
            font-family: 'Poppins', sans-serif;
        ">  
        
            <div style="text-align: center; display: flex; justify-content: center; align-items: center; gap: 10px;">
            <img src="https://i.ibb.co/VLwzcq2/logo.png" alt="Icon" style="width: 50px;">
                <h1 style="margin: 0; font-size: 28px;">
                    Welcome to <span style='color: #3275b8;'>Safe Navigation Speed Estimation</span>
                </h1>
            <img src="https://i.ibb.co/VLwzcq2/logo.png" alt="Icon" style="width: 50px;">
            </div>

            <p style="font-size: 20px; margin-top: 15px;">
                This application allows you to estimate the safe navigation speed for autonomous vehicles.
            </p>

            <ul style="font-size: 18px;">
                <li>🖥️ Uses stereo images to generate disparity and depth maps.</li>
                <li>🔍 Detects objects using advanced models like RT-DETRv2 and YOLOv11.</li>
                <li>🚦 Calculates safe distances for braking or obstacle avoidance.</li>
            </ul>

            <p style="font-weight: bold; font-size: 18px; margin-top: 20px;">
                👉 To get started, upload your stereo images or use the provided examples.
            </p>

            <!-- GIF animado -->
            <div style="text-align: center; margin-top: 20px;">
                <img src="https://i.ibb.co/TxBZCGQH/proccess.gif" alt="Near" style="width: 100%;">
            </div>
        </div>

        <style>
            @keyframes slideUpFadeIn {
                from {
                    opacity: 0;
                    transform: translateY(30px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');
        </style>
        """)


    gr.HTML("""
    <style>
        .title-text {
            font-family: 'Gill Sans Extrabold', sans-serif;
            font-size: 36px;
            font-weight: bold;
            color: #333;
            text-align: center; 
        }
        .description-text {
            font-family: 'Arial', sans-serif;
            font-size: 24px;
            color: #666;
            text-align: center;
            margin-top: 10px;
        }
            
        .welcome-image {
            display: block;
            margin: 0 auto;
            width: 60%;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
            
        .gradio-tabs {
            display: flex;
            justify-content: center;
        }
        .gradio-tab {
            display: flex;
            margin-bottom: 10px;
            justify-content: center;
        }


        
        @media (max-width: 768px) {
            .title-text {
                font-size: 24px;
            }
            .description-text {
                font-size: 18px;
            }
            .welcome-image {
                width: 80%;`
            }
            .gradio-tabs {
                display: flex;
                justify-content: center;
            }
            .gradio-tab {
                display: flex;
                margin-bottom: 10px;
                justify-content: center;
            }
        }
        @media (max-width: 480px) {
            .title-text {
                font-size: 18px;
            }
            .description-text {
                font-size: 14px;
            }
            .welcome-image {
                width: 100%;
            }
        }
    </style>
    """)

    gr.HTML("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500&display=swap');

        .title-text {
            text-align: center;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 10px; 
            flex-wrap: nowrap;
        }

        .title-text span {
            font-family: 'Poppins', sans-serif; /* Cambia la fuente a Poppins */
            font-size: 36px; /* Tamaño de fuente fijo */
            font-weight: 500; /* Peso de la fuente */
            color: #333; /* Cambia el color del texto si lo necesitas */
            text-align: center;
        }

        .title-text img {
            width: 40px;
            height: auto;
        }

        @media (max-width: 768px) {
            .title-text span {
                font-size: 20px; /* Ajusta el tamaño de la fuente en pantallas medianas */
            }
            .title-text img {
                width: 35px; /* Ajusta el tamaño del ícono */
            }
        }

        @media (max-width: 480px) {
            .title-text span {
                font-size: 16px; /* Ajusta el tamaño de la fuente en móviles */
            }
            .title-text img {
                width: 30px; /* Ajusta el tamaño del ícono */
            }
        }
    </style>
    <div class="title-text">
        <img src="https://i.ibb.co/VLwzcq2/logo.png" alt="Icon">
        <span>Estimation of safe navigation speed for autonomous vehicles</span>
        <img src="https://i.ibb.co/VLwzcq2/logo.png" alt="Icon">
    </div>
    """)


    with gr.Tab("Stereo Inference"):
        gr.Markdown("## Stereo Inference", elem_id="stereo-inference-title")
        gr.HTML("""
        <style>
            #stereo-inference-title {
                text-align: center;
            }
            .example-label {
                font-size: 20px;
                font-weight: bold;
                text-align: center;
                margin-top: 10px;
            }
                
            #inference-button {
                background-color: #2b3b46;
                color: white;
                border: none;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                margin: 4px 2px;
                cursor: pointer;
                border-radius: 8px;
            }
            #inference-button:hover {
                background-color: #688581;
            }
                
            @media (prefers-color-scheme: light) {
                #inference-button {
                    background-color: #e0e0e0;
                    color: #000000;
                }

                #inference-button:hover {
                    background-color: #C9D6D3
                }
            }    

           #depth-button {
                background-color: #2b3b46;
                color: white;
                border: none;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                margin: 4px 2px;
                cursor: pointer;
                border-radius: 8px;
            }
            #depth-button:hover {
                background-color: #C9D6D3;
            }
            @media (prefers-color-scheme: light) {
                #depth-button {
                    background-color: #e0e0e0;
                    color: #000000;
                }

                #depth-button:hover {
                    background-color: #C9D6D3; 
                }
            }
    
        </style>
        <p style="text-align: center;">Upload a pair of stereo images or choose the Example Stereo Images below, to perform stereo inference and generate the disparity map.</p>
        """)


        dataset_selection = gr.Radio(["Own images", "KITTI", "Driving Stereo"], label="Select Dataset", value="Own images", elem_id="model-selector")

        with gr.Row():
            image_path_left = gr.Image(label="Left Image")
            image_path_right = gr.Image(label="Right Image")


        with gr.Row():
            # Campo para mostrar los parámetros seleccionados
            focal_length_display = gr.Number(label="Focal Length [px]", interactive=True)
            baseline_display = gr.Number(label="Baseline [meters]", interactive=True)

        # Función para actualizar imágenes y parámetros según el dataset seleccionado
        def update_examples(selected_dataset):
            if selected_dataset == "KITTI":
                random_index = random.randint(0, len(KTTI_IMAGES_LEFT) - 1)
                return (
                    KTTI_IMAGES_LEFT[random_index],
                    KITTI_IMAGES_RIGHT[random_index],
                    KITTI_FOCAL_LENGTH, KITTI_BASELINE
                )
            elif selected_dataset == "Driving Stereo":
                random_index = random.randint(0, len(DRIVING_STEREO_IMAGES_LEFT) - 1)
                return (
                    DRIVING_STEREO_IMAGES_LEFT[random_index],
                    DRIVING_STEREO_IMAGES_RIGHT[random_index],
                    DRIVING_STEREO_FOCAL_LENGTH, DRIVING_STEREO_BASELINE
                )
            elif selected_dataset == "Own images":
                return (
                    None, None,
                    "", "",
                )

        dataset_selection.change(
            update_examples, 
            inputs=[dataset_selection], 
            outputs=[
                image_path_left, image_path_right,
                focal_length_display, baseline_display
            ]
        )

        run_button = gr.Button("Run Inference", elem_id="inference-button")
        output_image = gr.Image(label="Disparity Map", visible=True, scale=1, elem_id="disparity-map", height="auto", width="100%")
        run_button.click(stereo_inference, inputs=[image_path_left, image_path_right], outputs=output_image)

        generate_depth_button = gr.Button("Generate Depth Map", elem_id="depth-button", scale=1)
        depth_image = gr.Image(label="Depth Map", visible=True)

        # Conectar el botón con la función
        def display_depth_map(selected_dataset, focal_length, baseline):
            global stereo_output_image

            if stereo_output_image is None:
                return gr.Warning("Please run stereo inference first to generate the disparity map.")
            
            if selected_dataset == "Own images" and (not focal_length or not baseline):
                return gr.Warning("⚠️ Please provide valid focal length and baseline values for 'Own Images'.")
            
            focal_length, baseline = get_camera_parameters(selected_dataset, focal_length, baseline)

            if focal_length is None or baseline is None:
                return gr.Warning("Please provide the focal length and baseline values.")

            #print(f"✅ Parámetros recibidos en `display_depth_map`: Focal Length = {focal_length}, Baseline = {baseline}")

            # Generar el mapa de profundidad con la barra de colores
            depth_output_path_with_colorbar = generate_depth_map(stereo_output_image, focal_length, baseline)

            # Generar y guardar el mapa de profundidad sin visualizar
            only_depth_map(stereo_output_image, focal_length, baseline)

            # Devolver la imagen generada directamente
            return gr.update(value=depth_output_path_with_colorbar, visible=True)


        generate_depth_button.click(display_depth_map, inputs=[dataset_selection, focal_length_display, baseline_display], outputs=depth_image)

    with gr.Tab("Object Detection"):
        gr.Markdown("## Object Detection", elem_id="object-detection-title")
        gr.HTML("""
        <style>
            #object-detection-title {
                text-align: center;
            }
        </style>
                
        <p style="text-align: center;">Perform object detection on the original left image using the detected objects from the depth map. You can select the detection model</p>
        """)
        model_selector = gr.Radio(["RT-DETRv2", "YOLOv11"], label="Detection Model", value="RT-DETRv2", elem_id="model-selector")

        gr.HTML("""
        <style>
            #model-selector {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            margin: 0 auto;
            text-align: center;
            }
            #model-selector > label {
            text-align: center;
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 10px;
            }
            #model-selector .gr-button-group {
            display: flex;
            justify-content: center;
            gap: 10px; /* Spacing between buttons */
            }

            @media (max-width: 768px) {
            .gradio-row {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            }
            }

            input[type="radio"]:hover + label {
            background-color: #C7EDE4;
            color: #ffffff;
            }
        </style>
        """)

        run_button = gr.Button("Run Detection", elem_id="inference-button")
        detect_output_image = gr.Plot(label="Object Detection", visible=True, scale=1)
        cards_placeholder = gr.HTML(label="Detected Objects Info", visible=True)
    
    run_button.click(object_detection_with_disparity, inputs=[model_selector, dataset_selection, focal_length_display, baseline_display], outputs=[detect_output_image, cards_placeholder])

    with gr.Tab("Safe speed distance"):
        gr.Markdown("## Safe speed calculation section", elem_id="calculate-distance-title")
        gr.HTML("""
        <style>
            #calculate-distance-title {
                text-align: center;
            }
        </style>
                
        <p style="text-align: center;">Calculate the lookahead distance for stopping and swerving and optic system parameters based on the selected objects and vehicle parameters.</p>        
        """)        
        with gr.Row():
            with gr.Column():
                mu = gr.Slider(0.0, 1.0, value=0.8, step=0.01, label="Coefficient of friction (mu)")
                t = gr.Slider(0.0, 0.5, value=0.3, step=0.01, label="Perception time (t) [s]")
                l = gr.Slider(0.0, 0.5, value=0.25, step=0.01, label="Latency (l) [s]")
                B = gr.Slider(0.0, 3.0, value=2.0, step=0.1, label="Offset distance (df) [m]")

            with gr.Column():
                vehicle_name = gr.Dropdown(
                    label="Vehicle Model", 
                    choices=["Volkswagen Passat (B6)","Honda Aviancer","Tesla S", "Toyota Supra"],
                    interactive=True
                )
                turning_car = gr.Slider(0.0, 20.0, value=11.4, step=1.0, label="Turning Radius [m]")
                cog = gr.Slider(0.0, 2.0, value=0.55, step=0.01, label="Height of Center Gravity (COG) [m]")
                wheelbase = gr.Slider(0.0, 3.0, value=2.71, step=0.01, label="Width of Wheelbase [m]")

                vehicle_name.change(
                    fn=update_vehicle_params, 
                    inputs=[vehicle_name],
                    outputs=[turning_car, cog, wheelbase]
                )
        
        load_button = gr.Button("Load Objects", elem_id="inference-button")

        # Radio para seleccionar los objetos detectados
        selected_object_ids = gr.Radio(label="Select Object by ID", choices=[], interactive=True)
        
        run_button = gr.Button("Calculate Distance", elem_id="inference-button")
        
        # Organizar las gráficas en dos filas
        with gr.Row():
            with gr.Column():
                distance_plot1 = gr.Plot(label="Angle of View (AOV)")
                distance_plot3 = gr.Plot(label="Safe navigation speed for Stopping")
            with gr.Column():
                distance_plot2 = gr.Plot(label="Obstacle IFOV")
                distance_plot4 = gr.Plot(label="Safe navigation speed for Swerving")
            
        def calculate_distance_and_save(mu, t, l, B, turning_car, cog, wheelbase, selected_object_id):
            try:
                results = calculate_distance(mu, t, l, B, turning_car, cog, wheelbase, selected_object_id)

                # Extraer y manejar todas las variables correctamente
                (
                    fig1, fig2, fig3, fig4,
                    selected_ifov, 
                    safe_speed_stop, safe_speed_swerve, 
                    safe_speed_stop_haov, safe_speed_swerve_haov,
                    safe_speed_stop_vaov, safe_speed_swerve_vaov,
                    object_height_meters
                ) = results

                # Guardar los resultados en un archivo
                with open("./outputs/distance_calculation_results.txt", "w") as file:
                    file.write(f"✅ Calculation Results\n\n")
                    file.write(f"Selected Object ID: {selected_object_id}\n")
                    file.write(f"Coefficient of friction (mu): {mu}\n")
                    file.write(f"Perception time (t): {t} s\n")
                    file.write(f"Latency (l): {l} s\n")
                    file.write(f"Offset distance (df): {B} m\n")
                    file.write(f"Turning Radius: {turning_car} m\n")
                    file.write(f"Height of Center Gravity (COG): {cog} m\n")
                    file.write(f"Width of Wheelbase: {wheelbase} m\n")
                    file.write("\n")

                    file.write(f"Object distance:{object_distance} m\n")

                    # Guardar las variables de cotas
                    file.write("Cotas de seguridad:\n")
                    file.write(f" - Selected IFOV (degrees): {selected_ifov * (180/np.pi):.2f}\n" if selected_ifov else " - Selected IFOV: N/A\n")
                    file.write(f" - Safe Speed Stop (km/h): {safe_speed_stop:.2f}\n" if safe_speed_stop else " - Safe Speed Stop: N/A\n")
                    file.write(f" - Safe Speed Swerve (km/h): {safe_speed_swerve:.2f}\n" if safe_speed_swerve else " - Safe Speed Swerve: N/A\n")

                    file.write("\nCotas en HAOV y VAOV:\n")
                    file.write(f" - Safe Speed Stop (HAOV): {safe_speed_stop_haov:.2f} degrees\n" if safe_speed_stop_haov else " - Safe Speed Stop (HAOV): N/A\n")
                    #file.write(f" - Safe Speed Swerve (HAOV): {safe_speed_swerve_haov:.2f} degrees\n" if safe_speed_swerve_haov else " - Safe Speed Swerve (HAOV): N/A\n")
                    file.write(f" - Safe Speed Stop (VAOV): {safe_speed_stop_vaov:.2f} degrees\n" if safe_speed_stop_vaov else " - Safe Speed Stop (VAOV): N/A\n")
                    #file.write(f" - Safe Speed Swerve (VAOV): {safe_speed_swerve_vaov:.2f} degrees\n" if safe_speed_swerve_vaov else " - Safe Speed Swerve (VAOV): N/A\n")

                    # Guardar la altura del objeto
                    file.write(f"\nObject Height (meters): {object_height_meters:.4f} m\n")

                    file.write("\nGraphs:\n")
                    file.write("1. Angle of View (AOV)\n")
                    file.write("2. Obstacle IFOV\n")
                    file.write("3. Safe navigation speed for Stopping\n")
                    file.write("4. Safe navigation speed for Swerving\n")

                #print(f"✅ Results successfully saved to: ./outputs/distance_calculation_results.txt")

                return fig1, fig2, fig3, fig4, selected_ifov, safe_speed_stop, safe_speed_swerve, safe_speed_stop_haov, safe_speed_swerve_haov, safe_speed_stop_vaov, safe_speed_swerve_vaov, object_height_meters

            except ValueError as e:
                return gr.Warning(f"! Error during calculation: {str(e)}"), None, None, None


        run_button.click(calculate_distance_and_save, inputs=[mu, t, l, B, turning_car, cog, wheelbase, selected_object_ids], outputs=[distance_plot1, distance_plot2, distance_plot3, distance_plot4])

        # Acción del botón "Load Object Heights" para actualizar las opciones del Radio
        load_button.click(
            fn=lambda: gr.update(choices=[obj['id'] for obj in objects_info]) if objects_info else gr.Warning("No detected objects found."),
            inputs=None,
            outputs=selected_object_ids,
        )

demo.launch(share=True)

gr.HTML("""
<style>
    body {
        margin: 0;
        padding: 0;
        font-family: 'Arial', sans-serif;
        transition: background-color 0.3s ease, color 0.3s ease;
    }

    /* Estilo por defecto - oscuro */
    body {
        background-color: #2c2f33;
        color: #ffffff;
    }

    /* Estilo para modo claro */
    @media (prefers-color-scheme: light) {
        body {
            background-color: #ffffff;
            color: #000000;
        }

        .card {
            background-color: #f8f9fa;
            color: #000000;
        }

        .card h3 {
            color: #333333;
        }

        .card p {
            color: #555555;
        }
    }

    /* Estilo general para texto */
    h1, h2, h3, h4, h5, h6, p, label, .description-text {
        transition: color 0.3s ease;
    }

    @media (prefers-color-scheme: light) {
        h1, h2, h3, h4, h5, h6, p, label, .description-text {
            color: #000000;
        }
    }

    /* Estilo de botones */
    button {
        background-color: #1abc9c;
        color: #ffffff;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
        transition: background-color 0.3s ease, color 0.3s ease;
    }

    button:hover {
        background-color: #16a085;
    }

    @media (prefers-color-scheme: light) {
        button {
            background-color: #007bff;
            color: #ffffff;
        }

        button:hover {
            background-color: #0056b3;
        }
    }

    /* Estilo de las tarjetas */
    .card {
        background-color: #2c2f33;
        color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        text-align: left;
        transition: background-color 0.3s ease, color 0.3s ease;
    }

    @media (prefers-color-scheme: light) {
        .card {
            background-color: #f8f9fa;
            color: #000000;
        }
    }

    /* Estilo de enlaces */
    a {
        color: #1abc9c;
        transition: color 0.3s ease;
    }

    a:hover {
        color: #16a085;
    }

    @media (prefers-color-scheme: light) {
        a {
            color: #007bff;
        }

        a:hover {
            color: #0056b3;
        }
    }

    /* Estilo de tablas */
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
        font-size: 18px;
        text-align: left;
        transition: background-color 0.3s ease, color 0.3s ease;
    }

    th, td {
        padding: 12px;
        border: 1px solid #ddd;
    }

    th {
        background-color: #4CAF50;
        color: white;
    }

    @media (prefers-color-scheme: light) {
        th {
            background-color: #007bff;
            color: white;
        }

        td {
            color: #000000;
        }
    }

    /* Estilo de inputs */
    input, select, textarea {
        background-color: #3a3f44;
        color: #ffffff;
        border: 1px solid #555555;
        padding: 10px;
        border-radius: 5px;
        transition: background-color 0.3s ease, color 0.3s ease;
    }

    @media (prefers-color-scheme: light) {
        input, select, textarea {
            background-color: #ffffff;
            color: #000000;
            border: 1px solid #cccccc;
        }
    }

    /* Estilo para gráficos */
    .plotly-graph {
        transition: background-color 0.3s ease;
    }

    @media (prefers-color-scheme: light) {
        .plotly-graph {
            background-color: #ffffff;
        }
    }
</style>
""")
