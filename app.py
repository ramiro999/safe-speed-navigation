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

# Añadir directorios al path
sys.path.extend([
    os.path.join(os.path.dirname(__file__), 'stereo_estimation', 'NMRF'),
    os.path.join(os.path.dirname(__file__), 'stereo_estimation', 'NMRF', 'ops', 'setup', 'MultiScaleDeformableAttention')
])

# Importar módulos personalizados
from safety_calculator import calculate_lookahead_distance
from detr.image_processing import preprocess_image
from detr.model_detr import load_detr_model, COCO_INSTANCE_CATEGORY_NAMES
from yolov11.model_yolo import load_yolov11_model
from stereo_estimation.NMRF.disparity_inference import run_inference

"""
Función main para la interfaz de Gradio.
"""

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

# Cargar el modelo DETR
detr_model = load_detr_model()

# Cargar el modelo YOLOv11
yolo_model_path = "./yolov11/model/yolo11s.pt"
yolov11_model = load_yolov11_model(yolo_model_path) 

# Variable para el modelo seleccionado
selected_model = "DETR"  # Esto se actualizará según la selección del usuario

# Variable global para almacenar la imagen de salida de Stereo Inference
stereo_output_image = None
stereo_output_pred = None
image_path_left_original = None
objects_info = []  # Almacenar los objetos detectados en la imagen

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

        # Guardar el disp_pred como archivo .npy
        disp_pred_path = os.path.join(output, "disp_pred.npy")
        np.save(disp_pred_path, disp_preds[0]) # Guardar el primer disp_pred

        stereo_output_pred = disp_preds[0]  # Guardar la matriz de disparidad en la variable global

        print(f"Disparity prediction saved at: {disp_pred_path}")

        print(f"Rangos minimo y maximo del disp_pred: {disp_preds[0].min()}, {disp_preds[0].max()}")

        # Mostrar la imagen de disparidad con un colorbar
        disparity_image = cv2.imread(stereo_output_image, cv2.IMREAD_GRAYSCALE)
        plt.figure(figsize=(10, 5))
        plt.imshow(disparity_image)
        plt.colorbar(label='Disparity (pixels)')
        plt.axis('off')

        # Guardar la imagen con el colorbar
        #disparity_with_colorbar_path = stereo_output_image.replace(".png", "_with_colorbar.png")
        #plt.savefig(disparity_with_colorbar_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        return gr.update(value=stereo_output_image, visible=True)

    # En caso de que no se encuentre un archivo, lanzar un error
    raise FileNotFoundError("No generated image was found in the results folder.")

def generate_depth_map(disparity_path=None, focal_length=725.0087, baseline=0.532725):
    """
    Genera un mapa de profundidad con una barra de colores en el rango de metros 0-100.
    """
    global stereo_output_pred

    if stereo_output_pred is None:
        raise ValueError("No disparity data available.")

    # Usar la matriz de disparidad global si no se proporciona una ruta
    disparity = stereo_output_pred

    #disparity = disparity.astype(np.float32) / 256.0

    #print(f"Valor minimo y maximo despues de dividir en 256 en la funcion generate depth map",disparity.min(), disparity.max())

    # Ajustar el rango de disparidad a 2-192 
    #disparity = np.clip(disparity, a_min=2, a_max=192)

    #print(f"Valor minimo y maximo despues de usar np.clip en la funcion generate depth map",disparity.min(), disparity.max())
    # Convertir a float32 para cálculos precisos
    # disparity = disparity.astype(np.float32)

    # Evitar divisiones por cero y valores muy pequeños
    min_disparity = 1e-8
    disparity[disparity < min_disparity] = min_disparity

    # Calcular el mapa de profundidad
    depth_map = (focal_length * baseline) / disparity

    depth_map = np.round(depth_map.numpy() * 256).astype(np.uint16)

    depth_map = depth_map.astype(np.float32) / 256.0


    print(f"Valor minimo y maximo del depth map en la funcion generate depth map",depth_map.min(), depth_map.max())

    # Aplicar límites razonables a la profundidad en metros
    # depth_map[depth_map > 100] = 100
    # depth_map[depth_map < 0] = 0

    # Crear figura con barra de colores
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300, frameon=False)
    im = ax.imshow(depth_map, cmap='inferno_r')
    
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', label='Depth (meters)')
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

    return output_path_with_colorbar

def only_depth_map(disparity_path=None, focal_length=725.0087, baseline=0.532725, input_image_path=None):
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

    # Usar la matriz de disparidad global si no se proporciona una ruta
    disparity = stereo_output_pred

    print(f"Valores mínimos y máximos en la función only_depth_map: {disparity.min()}, {disparity.max()}, {disparity.shape}, {disparity.dtype}")

    # Evitar divisiones por cero y valores muy pequeños
    min_disparity = 1e-8
    disparity[disparity < min_disparity] = min_disparity

    # Calcular el mapa de profundidad
    depth_map = (focal_length * baseline) / disparity

    print(f"Valores mínimos y máximos del depth map en la función only_depth_map: {depth_map.min()}, {depth_map.max()}")
    print(f"Forma del depth map en la función only_depth_map: {depth_map.shape}, {depth_map.dtype}")

    base_name = os.path.splitext(os.path.basename(stereo_output_image))[0]
    output_name = f"{base_name}.png"

    # Ruta de salida
    output_dir = "./outputs/depth_results"
    output_path = os.path.join(output_dir, output_name)

    # Guardar el mapa de profundidad como imagen en uint16
    save_depth_map = np.round(depth_map.numpy() * 256).astype(np.uint16)
    Image.fromarray(save_depth_map, mode='I;16').save(output_path)

    print(f"Mapa de profundidad guardado en: {output_path}")

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

# Función para procesar todas las imágenes en una carpeta (Disparidad)
def process_stereo_images(left_images, right_images, disparity_output_folder):
    os.makedirs(disparity_output_folder, exist_ok=True)

    clear_folder(disparity_output_folder)

    if len(left_images) != len(right_images):
        return gr.Warning("El número de imágenes izquierda y derecha no coincide.")

    processed_images = []
    for left, right in zip(left_images, right_images):
        # Obtener nombre base
        base_name = os.path.basename(left.name)

        # Ejecutar inferencia estéreo
        stereo_inference(left.name, right.name)

        # Guardar imagen de disparidad generada
        output_path = os.path.join(disparity_output_folder, base_name)
        cv2.imwrite(output_path, cv2.imread(stereo_output_image))
        processed_images.append(output_path)

    return processed_images[0]  # Solo muestra la primera imagen


def process_depth_images(disparity_npy_files, depth_output_folder, focal_length=725.0087, baseline=0.532725):
    """
    Procesa todas las matrices de disparidad en formato `.npy` y genera mapas de profundidad.
    """
    os.makedirs(depth_output_folder, exist_ok=True)

    clear_folder(depth_output_folder)

    processed_images = []
    for disparity_npy in disparity_npy_files:
        base_name = os.path.basename(disparity_npy.name).replace(".npy", ".png")

        # 🔹 Cargar el archivo `.npy`
        try:
            disparity = np.load(disparity_npy.name)
        except Exception as e:
            print(f"❌ Error cargando {disparity_npy.name}: {e}")
            continue

        if disparity is None or disparity.size == 0:
            print(f"❌ Error: Archivo vacío o no válido {disparity_npy.name}")
            continue

        # 🔹 Evitar división por cero
        min_disparity = 1e-8
        disparity[disparity < min_disparity] = min_disparity

        # 🔹 Calcular el mapa de profundidad
        depth_map = (focal_length * baseline) / disparity

        # 🔹 Convertir a uint16 para guardar correctamente
        save_depth_map = np.round(depth_map * 256).astype(np.uint16)
        output_path = os.path.join(depth_output_folder, base_name)
        Image.fromarray(save_depth_map, mode='I;16').save(output_path)

        print(f"✅ Mapa de profundidad guardado en: {output_path}")
        processed_images.append(output_path)

    return processed_images[0] if processed_images else gr.Warning("❌ No se generaron mapas de profundidad.")


# Función para detección de objetos y calcular la distancia usando la disparidad
def object_detection_with_disparity(selected_model_name):
    global stereo_output_image, image_path_left_original, objects_info
    if not stereo_output_image:
        return None, gr.Warning("A Stereo Inference output image has not been generated.")

    # Generar el mapa de profundidad Z (en metros)
    depth, depth_map_path = only_depth_map()
    image = cv2.imread(image_path_left_original, cv2.IMREAD_COLOR)

    detr_bboxes, yolo_bboxes = [], []  # Inicializar variables para las cajas delimitadoras

    # Seleccionar las clases según el modelo
    if selected_model_name == "DETR":
        class_names = COCO_INSTANCE_CATEGORY_NAMES
    elif selected_model_name == "YOLOv11":
        class_names = list(yolov11_model.model.names.values())
    else:
        return None, gr.Warning("Modelo no válido seleccionado.")

    bboxes, labels = [], []  # Inicializar variables comunes

    if selected_model_name == "DETR":
        # Procesar con DETR
        _, image_tensor = preprocess_image(image)
        with torch.no_grad():
            outputs = detr_model(image_tensor)
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.8
        bboxes = outputs['pred_boxes'][0, keep].numpy()
        labels = probas[keep].argmax(-1).numpy()
        detr_bboxes = outputs['pred_boxes'][0, keep].numpy()

    elif selected_model_name == "YOLOv11":
        # Procesar con YOLOv11
        results = yolov11_model(image_path_left_original)
        bboxes = [box.xyxy[0].tolist() for box in results[0].boxes]
        labels = [int(box.cls[0]) for box in results[0].boxes]
        yolo_bboxes = [box.xyxy[0].tolist() for box in results[0].boxes]
    else:
        return None, gr.Warning("Invalid model selected.")

    # Procesar información de objetos detectados
    objects_info = []

    for idx, (bbox, label) in enumerate(zip(bboxes, labels), start=1):
        if selected_model_name == "DETR":
            if label < 0 or label >= len(class_names):
                print(f"Invalid label detected: {label}")
                continue
            cx, cy, w, h = bbox
            x0, y0 = int((cx - w / 2) * image.shape[1]), int((cy - h / 2) * image.shape[0])
            x1, y1 = int((cx + w / 2) * image.shape[1]), int((cy + h / 2) * image.shape[0])
        elif selected_model_name == "YOLOv11":
            x0, y0, x1, y1 = map(int, bbox)

        height_bb = abs(y1 - y0)

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
        }
    </style>
    <div class="grid-container">
    """

    CLASS_IMAGES = {
        'car': 'https://img.icons8.com/color/48/000000/car--v1.png',
        'traffic light': 'https://i.ibb.co/KG6PrDH/icons8-traffic-lights-sign-96.png',
        'person': 'https://i.ibb.co/3zFHGg2/icons8-person-96-2.png',
        'bicycle': 'https://i.ibb.co/d4tnFHP/icons8-bicycle-100.png',
        'bus': 'https://i.ibb.co/0f20RxR/icons8-bus-96.png',
        'truck': 'https://i.ibb.co/HD53Wtq/icons8-truck-96.png'
    }

    def get_class_icon(class_name):
        return CLASS_IMAGES.get(class_name, 'https://img.icons8.com/color/48/000000/car--v1.png')

    for obj in objects_info:
        cards_html += f"""
        <div class="card">
            <h3>
                <img src={get_class_icon(obj['class'])} alt="Object Icon">
                Object ID: {obj['id']}
            </h3>
            <p><strong>Class:</strong> {obj['class']}</p>
            <p><strong>Height:</strong> {obj['height']} pixels</p>
            <p><strong>Coordinates:</strong> ({obj['bbox'][0]}, {obj['bbox'][1]}) to ({obj['bbox'][2]}, {obj['bbox'][3]})</p>
            <p><strong>Distance:</strong> {obj['distance']:.2f} meters</p>
        </div>
        """
    cards_html += "</div>"

    # Visualizar y guardar resultados en Plotly
    fig = go.Figure()

    # Añadir la imagen original al gráfico
    fig.add_trace(go.Image(z=image, colormodel='rgb'))

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
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        showlegend=False,
        autosize=True,
        margin=dict(t=0, b=40, l=0, r=0),
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
    object_height = obj['height']
    object_distance = obj['distance']

    if object_distance is None or object_distance == float('inf'):
        return gr.Warning("⚠️ Invalid object distance, please select another object."), None, None, None

    # Calcular las distancias de seguridad
    try:
        fig1, fig2, fig3, fig4 = calculate_lookahead_distance(
            mu=mu, t=t, l=l, B=B, cog=cog, wheelbase=wheelbase, 
            turning_angle=turning_car, object_height=object_height, 
            object_distance=object_distance, image_path=None
        )
        return fig1, fig2, fig3, fig4
    except Exception as e:
        return gr.Warning(f"Error during calculation: {str(e)}"), None, None, None


# Función para actualizar los parámetros del vehículo según el modelo seleccionado
def update_vehicle_params(vehicle_model):
    vehicle_params = {
        "Volkswagen Passat (B6)": (11.4, 0.55, 2.71),
        "Tesla S": (11.8, 0.46, 2.96),
        "Toyota Supra": (10.40, 0.4953, 2.47),
        "Ford Mustang Shelby GT350": (12.67, 0.4953, 2.72)
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

# Interfaz de Gradio con pestañas
with gr.Blocks(theme=seafoam) as demo:
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
        
        @media (max-width: 768px) {
            .title-text {
                font-size: 24px;
            }
            .description-text {
                font-size: 18px;
            }
            .welcome-image {
                width: 80%;
            }
            .gradio-tabs {
                display: block;
            }
            .gradio-tab {
                display: block;
                margin-bottom: 10px;
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


    with gr.Tab("Stereo Inference", elem_id="stereo-inference-tab"):
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
                    background-color: #bdbdbd;
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
                background-color: #688581;
            }
            @media (prefers-color-scheme: light) {
                #depth-button {
                    background-color: #e0e0e0;
                    color: #000000;
                }

                #depth-button:hover {
                    background-color: #bdbdbd; 
                }
            }
    
        </style>
        <p style="text-align: center;">Upload a pair of stereo images or choose the Example Stereo Images below, to perform stereo inference and generate the disparity map.</p>
        """)

        with gr.Tab("Generar Disparidad"):
            left_images = gr.File(label="Cargar imágenes izquierda", file_count="directory")
            right_images = gr.File(label="Cargar imágenes derecha", file_count="directory")
            disparity_output_folder = gr.Textbox(value="./outputs/disparity_results", label="Carpeta de salida para disparidad")

            disparity_button = gr.Button("Generar Mapas de Disparidad")
            disparity_output_image = gr.Image(label="Ejemplo de Mapa de Disparidad Generado")

            disparity_button.click(
                process_stereo_images,
                inputs=[left_images, right_images, disparity_output_folder],
                outputs=disparity_output_image
            )

        with gr.Tab("Generar Profundidad"):
            disparity_images = gr.File(label="Cargar mapas de disparidad", file_count="directory")
            depth_output_folder = gr.Textbox(value="./outputs/depth_results", label="Carpeta de salida para profundidad")

            depth_button = gr.Button("Generar Mapas de Profundidad")
            depth_output_image = gr.Image(label="Ejemplo de Mapa de Profundidad Generado")

            depth_button.click(
                process_depth_images,
                inputs=[disparity_images, depth_output_folder],
                outputs=depth_output_image
            )

        #demo.launch(share=True)

        with gr.Row():
            image_path_left = gr.Image(label="Left Image")
            image_path_right = gr.Image(label="Right Image")

        # Agregar ejemplos de imágenes estéreos 
        gr.HTML('<div class="example-label">Example Stereo Images</div>')
        
        examples = gr.Examples(
            examples=[
            ["./images/stereo_images/kitti_images_left/000013_10.png", "./images/stereo_images/kitti_images_right/000013_10.png"],
            ["./images/stereo_images/kitti_images_left/0000000030.png", "./images/stereo_images/kitti_images_right/0000000030.png"],
            ["./images/stereo_images/kitti_images_left/0000000045.png", "./images/stereo_images/kitti_images_right/0000000045.png"],
            ["./images/stereo_images/kitti_images_left/0000000060.png", "./images/stereo_images/kitti_images_right/0000000060.png"]
            ],
            inputs=[image_path_left, image_path_right]
        )

        run_button = gr.Button("Run Inference", elem_id="inference-button")
        output_image = gr.Image(label="Disparity Map", visible=True, scale=1, elem_id="disparity-map")
        run_button.click(stereo_inference, inputs=[image_path_left, image_path_right], outputs=output_image)

        generate_depth_button = gr.Button("Generate Depth Map", elem_id="depth-button")
        depth_image = gr.Image(label="Depth Map", visible=True)

        # Conectar el botón con la función
        def display_depth_map():
            global stereo_output_image
            if stereo_output_image is None:
                return gr.Warning("Please run stereo inference first to generate the disparity map.")

            # Generar el mapa de profundidad con la barra de colores
            depth_output_path_with_colorbar = generate_depth_map(stereo_output_image)

            # Generar y guardar el mapa de profundidad sin visualizar
            only_depth_map(stereo_output_image)

            # Devolver la imagen generada directamente
            return gr.update(value=depth_output_path_with_colorbar, visible=True)


        generate_depth_button.click(display_depth_map, inputs=[], outputs=depth_image)

    with gr.Tab("Object Detection"):
        gr.Markdown("## Object Detection", elem_id="object-detection-title")
        gr.HTML("""
        <style>
            #object-detection-title {
                text-align: center;
            }
        </style>
                
        <p style="text-align: center;">Perform object detection on the original left image using the detected objects from the disparity map. You can select the detection model</p>
        """)
        model_selector = gr.Radio(["DETR", "YOLOv11"], label="Detection Model", value="DETR", elem_id="model-selector")

        gr.HTML("""
        <style>
            #model-selector {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                margin: 0 auto;
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
                gap: 10px; /* Espaciado entre botones */
            }
        </style>
        """)

        run_button = gr.Button("Run Detection", elem_id="inference-button")
        detect_output_image = gr.Plot(label="Object Detection", visible=True)
        cards_placeholder = gr.HTML(label="Detected Objects Info", visible=True)
    
    run_button.click(object_detection_with_disparity, inputs=[model_selector], outputs=[detect_output_image, cards_placeholder])

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
                mu = gr.Slider(0.0, 1.0, value=0.3, step=0.01, label="Coefficient of friction (mu)")
                t = gr.Slider(0.0, 5.0, value=0.2, step=0.01, label="Perception time (t) [s]")
                l = gr.Slider(0.0, 5.0, value=0.25, step=0.01, label="Latency (l) [s]")
                B = gr.Slider(0.0, 3.0, value=2.0, step=0.1, label="Offset distance [m]")

            with gr.Column():
                vehicle_name = gr.Dropdown(
                    label="Vehicle Model", 
                    choices=["Volkswagen Passat (B6)","Tesla S", "Toyota Supra", "Ford Mustang Shelby GT350"],
                    interactive=True
                )
                turning_car = gr.Slider(0.0, 20.0, value=11.4, step=1.0, label="Turning Car [°]")
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
                distance_plot3 = gr.Plot(label="Lookahead Distance for Stopping")
            with gr.Column():
                distance_plot2 = gr.Plot(label="Obstacle IFOV")
                distance_plot4 = gr.Plot(label="Lookahead Distance for Swerving")
                
        # Conexión la función calculate_distance con los componentes de entrada/salida
        run_button.click(calculate_distance, inputs=[mu, t, l, B, turning_car, cog, wheelbase, selected_object_ids], outputs=[distance_plot1, distance_plot2, distance_plot3, distance_plot4])

        # Acción del botón "Load Object Heights" para actualizar las opciones del Radio
        load_button.click(
            fn=lambda: gr.update(choices=[obj['id'] for obj in objects_info]) if objects_info else gr.Warning("No detected objects found."),
            inputs=None,
            outputs=selected_object_ids,
        )
    
demo.launch(share=True)