#app.py

import sys
import os
import torch
import gradio as gr
import cv2
import numpy as np
import tempfile
from typing import Union, Iterable
from gradio.themes.utils import colors, fonts, sizes
from gradio.themes.base import Base

# Añadir directorios al path
sys.path.append('/home/ramiro-avila/simulation-gradio/stereo/NMRF')
sys.path.append('/home/ramiro-avila/simulation-gradio/stereo/NMRF/ops/setup/MultiScaleDeformableAttention')

# Importar módulos personalizados
from lookahead_calculator import calculate_lookahead_distance
from detr.image_processing import preprocess_image, plot_detr_results_with_distance
from detr.model_loader import load_detr_model, COCO_INSTANCE_CATEGORY_NAMES
from stereo.NMRF.inference import run_inference
from stereo.NMRF.nmrf.utils.frame_utils import readDepthVKITTI

# Cargar el modelo DETR
model = load_detr_model()

# Variable global para almacenar la imagen de salida de Stereo Inference
stereo_output_image = None
image_path_left_original = None
objects_info = []  # Almacenar los objetos detectados

def save_temp_image(image_array):
    """
    Guarda una imagen numpy.ndarray en un archivo temporal y devuelve la ruta del archivo.
    """
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    cv2.imwrite(temp_file.name, image_array)
    return temp_file.name

# Función stereo_inference()
def stereo_inference(image_path_left=None, image_path_right=None):
    """
    Realiza inferencia estéreo. Soporta tanto rutas de archivo como imágenes cargadas en memoria (numpy.ndarray).
    """
    # Verificar si las entradas son numpy.ndarray y guardar en archivos temporales si es necesario
    if isinstance(image_path_left, np.ndarray):
        image_path_left = save_temp_image(image_path_left)
    if isinstance(image_path_right, np.ndarray):
        image_path_right = save_temp_image(image_path_right)

    # Asegurarse de que las entradas son rutas válidas
    if not isinstance(image_path_left, str) or not isinstance(image_path_right, str):
        raise ValueError("Las entradas deben ser rutas de archivo o imágenes numpy.ndarray.")

    global stereo_output_image, image_path_left_original
    image_path_left_original = image_path_left  # Guardar la imagen original para la detección de objetos

    dataset_name = "custom_dataset"
    output = "./resultados_kitti"
    resume_path = "./stereo/NMRF/pretrained/kitti.pth"

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
    raise FileNotFoundError("No se encontró ninguna imagen generada en la carpeta de resultados.")

# Función para detección de objetos y calcular la distancia usando la disparidad
def object_detection_with_disparity():
    global stereo_output_image, image_path_left_original, objects_info
    if stereo_output_image is None or image_path_left_original is None:
        raise ValueError("No se ha generado una imagen de salida de Stereo Inference o no se ha proporcionado la imagen original.")

    # Leer la imagen de disparidad
    disparity, valid = readDepthVKITTI(stereo_output_image)

    # Leer la imagen RGB original (una de las imágenes estéreo)
    image = cv2.imread(image_path_left_original, cv2.IMREAD_UNCHANGED)

    # Verificar si la imagen tiene un canal alfa (es decir, 4 canales: RGBA) y convertir a RGB
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    # Si la imagen está en escala de grises, convertir a RGB
    if len(image.shape) == 2 or image.shape[-1] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Preprocesar la imagen para la entrada del modelo DETR
    _, image_tensor = preprocess_image(image)

    with torch.no_grad():
        outputs = model(image_tensor)

    # Filtrar las predicciones con alta confianza
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.8

    bboxes = outputs['pred_boxes'][0, keep].numpy()
    labels = probas[keep].argmax(-1).numpy()

    # Procesar información de objetos detectados
    objects_info = []
    for idx, (bbox, label) in enumerate(zip(bboxes, labels), start=1):
        cx, cy, w, h = bbox
        x0, y0 = int((cx - w / 2) * image.shape[1]), int((cy - h / 2) * image.shape[0])
        x1, y1 = int((cx + w / 2) * image.shape[1]), int((cy + h / 2) * image.shape[0])
        height_bb = abs(y1 - y0)

        # Recortar región del mapa de disparidad correspondiente al bounding box
        bbox_disp = disparity[y0:y1, x0:x1]
        bbox_valid = valid[y0:y1, x0:x1]

        # Calcular distancia promedio en región válida
        if bbox_valid.any():
            avg_distance = bbox_disp[bbox_valid].mean() / 100  # Convertir a metros
        else:
            avg_distance = float('inf')  # Si no hay valores válidos

        # Agregar información del objeto detectado
        objects_info.append({
            'id': idx,
            'class': COCO_INSTANCE_CATEGORY_NAMES[label],
            'height': height_bb,
            'bbox': [x0, y0, x1, y1],
            'distance': avg_distance
        })

    # Visualizar y guardar resultados
    ids = [obj['id'] for obj in objects_info]
    distances = [obj['distance'] for obj in objects_info]
    fig_detr = plot_detr_results_with_distance(image, bboxes, labels, ids, distances)
    fig_detr.savefig("./outputs/object_detection_results_with_distances.png", bbox_inches="tight")

    # Crear texto de resultados
    info_text = "Objetos detectados:\n\n"
    for obj in objects_info:
        info_text += f"ID: {obj['id']}\n"
        info_text += f"Clase: {obj['class']}\n"
        info_text += f"Altura objeto: {obj['height']:,} píxeles\n"
        info_text += f"Coordenadas: ({obj['bbox'][0]}, {obj['bbox'][1]}) a ({obj['bbox'][2]}, {obj['bbox'][3]})\n"
        info_text += f"Distancia promedio: {obj['distance']:.2f} metros\n\n"

    return "./outputs/object_detection_results_with_distances.png", info_text

# Función para el cálculo de distancia (sin detección de objetos)
def calculate_distance(mu, t, l, B, turning_car, cog, wheelbase, selected_object_id):
    global objects_info
    
    # Obtener información del objeto seleccionado
    selected_object = next((obj for obj in objects_info if obj['id'] == selected_object_id), None)
    if selected_object is None:
        raise ValueError("El objeto seleccionado no es válido.")
    
    object_height = selected_object['height']
    object_distance = selected_object['distance']
    
    # Realizar cálculos utilizando la información del objeto seleccionado
    _, plot_path1, plot_path2, plot_path3, plot_path4 = calculate_lookahead_distance(
        mu=mu,
        t=t,
        l=l,
        B=B,
        cog=cog,
        wheelbase=wheelbase,
        turning_angle=turning_car,
        object_height=object_height,  # Añadir la altura del objeto
        object_distance=object_distance,  # Añadir la distancia del objeto
        image_path=None  # Si necesitas una imagen de referencia, pásala aquí
    )
    return (
        gr.update(value=plot_path1, visible=True),
        gr.update(value=plot_path2, visible=True),
        gr.update(value=plot_path3, visible=True),
        gr.update(value=plot_path4, visible=True)
    )

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

    with gr.Tab("Home"):
        gr.HTML("""
        <div class="title-text">
            Estimation of safe navigation speed for autonomous vehicles
        </div>
                
        <div style="display: flex; justify-content: center; align-items: center; margin-top:10px">
            <img src="https://i.ibb.co/sJrp7P4/portada.png" class="welcome-image"/>
        </div>
        """)

    with gr.Tab("Stereo Inference"):
        gr.Markdown("## Stereo Inference")
        with gr.Row():
            image_path_left = gr.Image(label="Left Image")
            image_path_right = gr.Image(label="Right Image")
        output_image = gr.Image(label="Output Image", visible=False)
        run_button = gr.Button("Run Inference")
        run_button.click(stereo_inference, inputs=[image_path_left, image_path_right], outputs=output_image)

    with gr.Tab("Object Detection"):
        gr.Markdown("## Object Detection")
        run_button = gr.Button("Run Detection")
        detect_output_image = gr.Image(label="Object Detection Output Image", visible=True)
        detect_output_text = gr.Textbox(label="Detected Objects Info", lines=10, interactive=False)
        run_button.click(object_detection_with_disparity, outputs=[detect_output_image, detect_output_text])

    with gr.Tab("Calculate Distance"):
        gr.Markdown("## Safe distance calculation section")
        with gr.Row():
            with gr.Column():
                mu = gr.Slider(0.0, 1.0, value=0.3, step=0.01, label="Coeficiente de fricción (mu)")
                t = gr.Slider(0.0, 5.0, value=0.2, step=0.01, label="Tiempo de percepción (t) [s]")
                l = gr.Slider(0.0, 5.0, value=0.25, step=0.01, label="Latencia (l) [s]")
                B = gr.Slider(0.0, 5.0, value=2.0, step=0.1, label="Buffer (B) [m]")
            with gr.Column():
                turning_car = gr.Slider(0.0, 360.0, value=180.0, step=1.0, label="Turning Car [°]")
                cog = gr.Slider(0.0, 2.0, value=1.0, step=0.01, label="Height of Center Gravity (COG) [m]")
                wheelbase = gr.Slider(0.0, 3.0, value=1.5, step=0.01, label="Width of Wheelbase [m]")
                selected_object_id = gr.Number(value=1, label="Selected Object ID", precision=0, interactive=True)

        # Organizar las gráficas en dos filas
        with gr.Row():
            with gr.Column():
                distance_plot1 = gr.Image(type="filepath", label="Gráfica de distancia segura", visible=False)
                distance_plot3 = gr.Image(type="filepath", label="Campo de visión Instantaneo Positivo [miliradianes]", visible=False)
            with gr.Column():
                distance_plot2 = gr.Image(type="filepath", label="Gráfica del ángulo de visión", visible=False)
                distance_plot4 = gr.Image(type="filepath", label="Campo de visión instantaneo para los obstáculos", visible=False)
                
        run_button = gr.Button("Calculate Distance")
        run_button.click(calculate_distance, inputs=[mu, t, l, B, turning_car, cog, wheelbase, selected_object_id], outputs=[distance_plot1, distance_plot2, distance_plot3, distance_plot4])

demo.launch(share=True)