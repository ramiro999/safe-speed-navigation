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

# Importar módulos personalizados
from autonomous_navigation_calculator import calculate_lookahead_distance
from detr.image_processing import preprocess_image, plot_detr_results_with_distance
from detr.model_loader import load_detr_model, COCO_INSTANCE_CATEGORY_NAMES
from yolov11.model_loader_yolo import load_yolov11_model
from yolov11.image_processing_yolo import draw_yolo_results

# Añadir directorios al path
sys.path.extend([
    os.path.join(os.path.dirname(__file__), 'stereo', 'NMRF'),
    os.path.join(os.path.dirname(__file__), 'stereo', 'NMRF', 'ops', 'setup', 'MultiScaleDeformableAttention')
])

from stereo.NMRF.inference import run_inference
from stereo.NMRF.nmrf.utils.frame_utils import readDepthVKITTI

"""
Función main para la interfaz de Gradio.
"""
 
# Cargar el modelo DETR
detr_model = load_detr_model()

# Cargar el modelo YOLOv11
yolo_model_path = "./yolov11/model/yolo11s.pt"
yolov11_model = load_yolov11_model(yolo_model_path)  # Carga el modelo YOLOv11

# Variable para el modelo seleccionado
selected_model = "DETR"  # Esto se actualizará según la selección del usuario

# Variable global para almacenar la imagen de salida de Stereo Inference
stereo_output_image = None
image_path_left_original = None
objects_info = []  # Almacenar los objetos detectados en la imagen


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


def save_temp_image(image_array):
    """
    Guarda una imagen numpy.ndarray en un archivo temporal y devuelve la ruta del archivo.
    """
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    cv2.imwrite(temp_file.name, image_array)
    return temp_file.name

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

    global stereo_output_image, image_path_left_original # Variables globales para almacenar la imagen de salida y la imagen original
    image_path_left_original = image_path_left 
    dataset_name = "kitti" 
    output = "./resultados_kitti"
    resume_path = "./stereo/NMRF/pretrained/sceneflow.pth" # Ruta del modelo pre-entrenado con sceneflow.pth

    # Crear una lista con las imágenes de entrada proporcionadas por el usuario
    image_list = [(image_path_left, image_path_right)]

    # Realizar la inferencia usando las imágenes proporcionadas
    run_inference(dataset_name, output, resume_path, image_list, show_attr="disparity")

    # Buscar la última imagen generada en la carpeta de resultados
    result_files = [os.path.join(output, f) for f in os.listdir(output) if f.endswith(".png")]
    if result_files:
        # Ordenar por fecha de modificación
        result_files.sort(key=os.path.getmtime, reverse=True)
        stereo_output_image = os.path.abspath(result_files[0])
        return gr.update(value=stereo_output_image, visible=True)

    # En caso de que no se encuentre un archivo, lanzar un error
    raise FileNotFoundError("No generated image was found in the results folder.")

def generate_depth_map(disparity_path, focal_length=725.0087, baseline=0.532725):
    """
    Genera un mapa de profundidad con una barra de colores en el rango 0-255.
    """
    # Cargar el mapa de disparidad
    disparity = cv2.imread(disparity_path, cv2.IMREAD_GRAYSCALE)

    if disparity is None:
        raise ValueError("No se pudo cargar el mapa de disparidad.")

    # Convertir a float32 para cálculos precisos
    disparity = disparity.astype(np.float32)

    # Evitar divisiones por cero y valores muy pequeños
    min_disparity = 0.1
    disparity[disparity < min_disparity] = min_disparity

    # Calcular el mapa de profundidad
    depth_map = (focal_length * baseline) / disparity

    # Aplicar límites razonables a la profundidad
    depth_map[depth_map > 100] = 100
    depth_map[depth_map < 0] = 0

    # Normalizar para visualización en el rango 0-255
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)

    # Crear figura con barra de colores
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300, frameon=False)
    im = ax.imshow(depth_map_normalized, cmap='inferno_r', vmin=0, vmax=255)
    
    # Agregar colorbar con rango 0-255
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', label='Depth')
    cbar.ax.xaxis.label.set_color('white')
    cbar.ax.tick_params(color='white')  
    cbar.ax.xaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color='white')  # Cambiar el color de las etiquetas a blanco
    cbar.set_ticks([0, 64, 128, 192, 255])  # Valores clave en el rango
    cbar.ax.set_xticklabels(['0', '64', '128', '192', '255'])  # Escala en el rango 0-255

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
    output_path_with_colorbar = "./outputs/depth_with_colorbar_0_255.png"
    plt.savefig(output_path_with_colorbar, bbox_inches='tight', pad_inches=0, dpi=300, transparent=True)
    plt.close()

    return output_path_with_colorbar

def only_depth_map(disparity_path, focal_length=725.0087, baseline=0.532725):
    """
    Genera un mapa de profundidad a partir de un mapa de disparidad.
    """
    # Cargar el mapa de disparidad
    disparity = cv2.imread(disparity_path, cv2.IMREAD_GRAYSCALE)
    
    if disparity is None:
        raise ValueError("No se pudo cargar el mapa de disparidad.")

    # Convertir a float32 para cálculos precisos
    disparity = disparity.astype(np.float32)
    
    # Normalizar la disparidad si está en el rango 0-255
    # Asumiendo que los valores máximos de disparidad están alrededor de 128-256 píxeles
    #disparity = disparity / 16.0  # Factor común en mapas de disparidad
    
    # Evitar divisiones por cero y valores muy pequeños
    min_disparity = 0.1
    disparity[disparity < min_disparity] = min_disparity

    # Calcular el mapa de profundidad
    depth_map = (focal_length * baseline) / disparity

    # Aplicar límites razonables a la profundidad
    depth_map[depth_map > 100] = 100  # Limitar a 100 metros
    depth_map[depth_map < 0] = 0

    # Normalizar para visualización
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_map_normalized = depth_map_normalized.astype(np.uint8)

    # Aplicar un mapa de color para mejor visualización
    depth_map_colored = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_INFERNO)

    return depth_map_colored

# Función para detección de objetos y calcular la distancia usando la disparidad
def object_detection_with_disparity(selected_model_name):
    global stereo_output_image, image_path_left_original, objects_info

    if not stereo_output_image:
        return None, gr.Warning("A Stereo Inference output image has not been generated.")

    # Generar el mapa de profundidad
    depth_map_colored = only_depth_map(stereo_output_image)
    depth = cv2.cvtColor(depth_map_colored, cv2.COLOR_BGR2GRAY)
    image = cv2.imread(image_path_left_original, cv2.IMREAD_COLOR)

    # Seleccionar las clases según el modelo
    if selected_model_name == "DETR":
        class_names = COCO_INSTANCE_CATEGORY_NAMES
    elif selected_model_name == "YOLOv11":
        class_names = yolov11_model.model.names
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

    elif selected_model_name == "YOLOv11":
        # Procesar con YOLOv11
        results = yolov11_model(image_path_left_original)
        bboxes = [box.xyxy[0].tolist() for box in results[0].boxes]
        labels = [int(box.cls[0]) for box in results[0].boxes]
    else:
        return None, gr.Warning("Invalid model selected.")

    # Procesar información de objetos detectados
    objects_info = []

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
            median_distance = np.median(bbox_valid) #* (100 / 255)  # Limitar rango de 0-100 metros para valores 0-255
        else:
            median_distance = float('inf')

        # Agregar información del objeto detectado
        objects_info.append({
            'id': idx,
            'class': class_names[label],
            'height': height_bb,
            'bbox': [x0, y0, x1, y1],
            'distance': median_distance
        })
        

        # Diccionario para las imagenes de las clases seleccionadas
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
       


        # Tarjeta para el objeto detectado
        cards_html += f"""
        <div class="card">
            <h3>
                <img src={get_class_icon(class_names[label])} alt="Object Icon">
                Object ID: {idx}
            </h3>
            <p><strong>Class:</strong> {class_names[label]}</p>
            <p><strong>Height:</strong> {height_bb} pixels</p>
            <p><strong>Coordinates:</strong> ({x0}, {y0}) to ({x1}, {y1})</p>
            <p><strong>Distance:</strong> {median_distance:.2f} meters</p>
        </div>
        """

    # Finalizar el contenedor de la grilla
    cards_html += "</div>"


    # Ordenar los objetos por distancia
    objects_info.sort(key=lambda x: x['distance'])
    
    # Reasignar los objetos ordenados
    for new_idx, obj in enumerate(objects_info, start=1):
        obj['id'] = new_idx 

    # Visualizar y guardar resultados en Plotly
    fig = go.Figure()

    # Añadir la imagen original al gráfico
    fig.add_trace(go.Image(
        z=image,
        colormodel='rgb'
    ))

    # Añadir las cajas delimitadoras sobre la imagen
    for obj in objects_info:
        x0, y0, x1, y1 = obj['bbox']
        color = px.colors.qualitative.Plotly[COCO_INSTANCE_CATEGORY_NAMES.index(obj['class']) % len(px.colors.qualitative.Plotly)]
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
    fig.write_image("./outputs/object_detection_results_with_distances.png")

    return fig, cards_html

# Función para el cálculo de distancia segura y gráficos de decisión
def calculate_distance(mu, t, l, B, turning_car, cog, wheelbase, selected_object_id):
    global objects_info

    if selected_object_id is None:
        return gr.Warning("Please select at least one detected object to calculate the distance.")

    # Ensure selected_object_id is a list
    selected_object_ids = [selected_object_id] if isinstance(selected_object_id, int) else selected_object_id

    # Filter selected objects
    selected_objects = [obj for obj in objects_info if obj['id'] in selected_object_ids]

    if not selected_objects:
        return gr.Warning("⚠️ Selected object ID not found in the detected objects.")
    
    # Verifica que las distancias de los objetos sean numéricas y válidas
    distances = [obj.get('distance', None) for obj in selected_objects if isinstance(obj.get('distance', None), (int, float))]
    distances = [dist for dist in distances if not (np.isinf(dist) or np.isnan(dist))]

    if not distances:
        return gr.Warning("⚠️ None of the selected objects have valid distances.")

    # Calculate average height and distance of selected objects
    avg_height = np.mean([obj['height'] for obj in selected_objects])
    avg_distance = np.mean(distances)

    # Calculate decision graph using lookahead distance function
    fig1, fig2, fig3, fig4 = calculate_lookahead_distance(
        mu=mu,
        t=t,
        l=l,
        B=B,
        cog=cog,
        wheelbase=wheelbase,
        turning_angle=turning_car,
        object_height=avg_height,
        object_distance=avg_distance,
        image_path=None
    )

    return fig1, fig2, fig3, fig4


# Función para actualizar los parámetros del vehículo según el modelo seleccionado
def update_vehicle_params(vehicle_model):
    vehicle_params = {
        "Volkswagen Passat (B6)": (11.4, 0.55, 2.71),
        "Tesla S": (11.8, 0.46, 2.96),
        "Toyota Supra": (10.40, 0.4953, 2.47),
        "Ford Mustang Shelby GT350": (12.67, 0.4953, 2.72)
    }
    return vehicle_params.get(vehicle_model, (0, 0, 0))

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
            margin-top: 15px;
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
            gap: 10px; /* Espacio entre íconos y texto */
            flex-wrap: nowrap; /* No permite que se rompa en varias líneas */
        }

        .title-text span {
            font-family: 'Poppins', sans-serif; /* Cambia la fuente a Poppins */
            font-size: 2.5vw; /* Escala adaptativa según el ancho de la pantalla */
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
                font-size: 2.5vw; /* Reduce la fuente en pantallas medianas */
            }
            .title-text img {
                width: 35px; /* Ajusta el tamaño del ícono */
            }
        }

        @media (max-width: 480px) {
            .title-text span {
                font-size: 3.5vw; /* Reduce aún más la fuente en móviles */
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

        with gr.Row():
            image_path_left = gr.Image(label="Left Image")
            image_path_right = gr.Image(label="Right Image")

        # Agregar ejemplos de imágenes estéreos 
        gr.HTML('<div class="example-label">Example Stereo Images</div>')
        examples = gr.Examples(
            examples=[
            ["./stereo_images/images_left/000005_10.png", "./stereo_images/images_right/000005_10.png"],
            ["./stereo_images/images_left/000070_11.png", "./stereo_images/images_right/000070_11.png"],
            ["./stereo_images/images_left/000179_11.png", "./stereo_images/images_right/000179_11.png"],
            ["./stereo_images/images_left/000194_11.png", "./stereo_images/images_right/000194_11.png"]
            ],
            inputs=[image_path_left, image_path_right]
        )
        #164_10


        run_button = gr.Button("Run Inference", elem_id="inference-button")
        output_image = gr.Image(label="Disparity Map", visible=True)
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
                turning_car = gr.Slider(0.0, 20.0, value=10.0, step=1.0, label="Turning Car [°]")
                cog = gr.Slider(0.0, 2.0, value=0.5, step=0.01, label="Height of Center Gravity (COG) [m]")
                wheelbase = gr.Slider(0.0, 3.0, value=1.5, step=0.01, label="Width of Wheelbase [m]")

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
                distance_plot2 = gr.Plot(label="Positive Obstacle IFOV")
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