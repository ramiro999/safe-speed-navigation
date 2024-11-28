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
from lookahead_calculator import generate_decision_graph

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
    cards_html = ""  # Variable para almacenar HTML de las tarjetas mejoradas
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

        # Tarjeta para el objeto detectado
        cards_html += f"""
        <div style="background-color: #2c2f33; color: #ffffff; border-radius: 10px; padding: 15px; margin-bottom: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
            <h3 style="margin-bottom: 10px; display: flex; align-items: center;">
                <img src="https://img.icons8.com/color/48/000000/car--v1.png" alt="Object Icon" style="width: 35px; height: 35px; margin-right: 10px;"> Object ID: {idx}
            </h3>
            <p style="margin: 5px 0;"><strong>Class:</strong> {COCO_INSTANCE_CATEGORY_NAMES[label]}</p>
            <p style="margin: 5px 0;"><strong>Height:</strong> {height_bb} pixels</p>
            <p style="margin: 5px 0;"><strong>Coordinates:</strong> ({x0}, {y0}) to ({x1}, {y1})</p>
            <p style="margin: 5px 0;"><strong>Average Distance:</strong> {avg_distance:.2f} meters</p>
        </div>
        """

    # Visualizar y guardar resultados
    ids = [obj['id'] for obj in objects_info]
    distances = [obj['distance'] for obj in objects_info]
    fig_detr = plot_detr_results_with_distance(image, bboxes, labels, ids, distances)
    fig_detr.savefig("./outputs/object_detection_results_with_distances.png", bbox_inches="tight")

    return "./outputs/object_detection_results_with_distances.png", cards_html


# Función para el cálculo de distancia utilizando gráficos interactivos de Plotly
def calculate_distance(mu, t, l, B, turning_car, cog, wheelbase, selected_object_ids):
    global objects_info

    # Asegurarse de que se seleccionen objetos
    if not selected_object_ids:
        raise ValueError("Debe seleccionar al menos un objeto para calcular la distancia.")

    # Filtrar los objetos seleccionados
    selected_objects = [obj for obj in objects_info if obj['id'] in selected_object_ids]
    if not selected_objects:
        raise ValueError("Ninguno de los objetos seleccionados es válido.")

    # Calcular la altura y distancia promedio de los objetos seleccionados
    avg_height = np.mean([obj['height'] for obj in selected_objects])
    avg_distance = np.mean([obj['distance'] for obj in selected_objects])

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


def update_vehicle_params(vehicle_model):
    if vehicle_model == "Tesla S":
        return 11.8, 0.46, 2.96
    elif vehicle_model == "Toyota Supra":
        return 10.40, 0.4953, 2.47
    elif vehicle_model == "Ford Mustang Shelby GT350":
        return 12.67, 0.4953, 2.72

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
    <div class="title-text" style="text-align: center; display: flex; justify-content: center; align-items: center;">
        <img src="https://i.ibb.co/v10hH1k/icons8-autonomous-vehicles-96.png" alt="Icon" style="width: 50px; vertical-align: middle; margin-right: 10px;">
        <span> Estimation of safe navigation speed for autonomous vehicles </span>
        <img src="https://i.ibb.co/v10hH1k/icons8-autonomous-vehicles-96.png" alt="Icon" style="width: 50px; vertical-align: middle; margin-left: 10px;">
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
        </style>
        <p style="text-align: center;">Upload a pair of stereo images or choose the Example Stereo Images below, to perform stereo inference and generate the disparity map.</p>
        """)

        with gr.Row():
            image_path_left = gr.Image(label="Left Image")
            image_path_right = gr.Image(label="Right Image")

        # Agregar ejemplos de imágenes estereoscópicas
        gr.HTML('<div class="example-label">Example Stereo Images</div>')
        examples = gr.Examples(
            examples=[
            ["./stereo_images/images_left/000106_11.png", "./stereo_images/images_right/000106_11.png"],
            ["./stereo_images/images_left/000070_11.png", "./stereo_images/images_right/000070_11.png"],
            ["./stereo_images/images_left/000108_10.png", "./stereo_images/images_right/000108_10.png"],
            ["./stereo_images/images_left/000194_11.png", "./stereo_images/images_right/000194_11.png"]
            ],
            inputs=[image_path_left, image_path_right]
        )

        run_button = gr.Button("Run Inference", elem_id="inference-button")
        output_image = gr.Image(label="Output Image", visible=True)
        run_button.click(stereo_inference, inputs=[image_path_left, image_path_right], outputs=output_image)

        gr.HTML("""
        <style>
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
            #run-inference-button:hover {
                background-color: #688581;
            }
        </style>
        """)

    with gr.Tab("Object Detection"):
        gr.Markdown("## Object Detection", elem_id="object-detection-title")
        gr.HTML("""
        <style>
            #object-detection-title {
                text-align: center;
            }
        </style>
                
        <p style="text-align: center;">Perform object detection on the original left image using the detected objects from the disparity map.</p>
        """)
        run_button = gr.Button("Run Detection", elem_id="inference-button")
        detect_output_image = gr.Image(label="Object Detection Output Image", visible=True)
        cards_placeholder = gr.HTML(label="Detected Objects Info", visible=True)  # Placeholder for object cards
    
    run_button.click(object_detection_with_disparity, outputs=[detect_output_image, cards_placeholder])

    with gr.Tab("Calculate Distance"):
        gr.Markdown("## Safe distance calculation section", elem_id="calculate-distance-title")
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
                B = gr.Slider(0.0, 5.0, value=2.0, step=0.1, label="Buffer distance (B) [m]")

            with gr.Column():
                vehicle_name = gr.Dropdown(
                    label="Vehicle Model", 
                    choices=["Tesla S", "Toyota Supra", "Ford Mustang Shelby GT350"],
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
        
        # Crear CheckboxGroup para seleccionar los objetos detectados
        selected_object_ids = gr.CheckboxGroup(label="Select Object(s) by ID", choices=[], interactive=True)    
        # Botón para cargar los objetos detectados
        load_button = gr.Button("Load Object Heights", elem_id="inference-button")
        
        run_button = gr.Button("Calculate Distance", elem_id="inference-button")
        
        # Organizar las gráficas en dos filas
        with gr.Row():
            with gr.Column():
                distance_plot1 = gr.Plot(label="Lookahead Distance for Stopping and Swerving")
                distance_plot3 = gr.Plot(label="IFOV")
            with gr.Column():
                distance_plot2 = gr.Plot(label="Angle of View (AOV)")
                distance_plot4 = gr.Plot(label="Positive Obstacle IFOV")
                
        # Conectar la función calculate_distance con los componentes de entrada/salida
        run_button.click(calculate_distance, inputs=[mu, t, l, B, turning_car, cog, wheelbase, selected_object_ids], outputs=[distance_plot1, distance_plot2, distance_plot3, distance_plot4])

        # Acción del botón "Load Object Heights" para actualizar las opciones del CheckboxGroup
        load_button.click(
            fn=lambda: gr.update(choices=[obj['id'] for obj in objects_info]),
            inputs=None,
            outputs=selected_object_ids,
        )

    with gr.Tab("Results"):
        gr.Markdown("## Result Making", elem_id="decision-making-title")
        gr.HTML("""
        <style>
            #decision-making-title {
                text-align: center;
            }
        </style>
                
        <p style="text-align: center;">Generate a decision graph for lookahead distance for Stopping And Swerving based on the selected objects and vehicle parameters.</p>        
                """)
        decision_plot = gr.Plot(label="Lookahead Distance for Stopping, Swerving, and Field of View (Decision)")

        # Botón para generar la gráfica de decisión
        decision_button = gr.Button("Generate Decision Graph", elem_id="inference-button")

        decision_button.click(
            lambda mu, t, l, B, turning_car, cog, wheelbase, selected_object_ids: generate_decision_graph(
                mu, t, l, B, turning_car, cog, wheelbase, selected_object_ids, objects_info
            ),
            inputs=[mu, t, l, B, turning_car, cog, wheelbase, selected_object_ids],
            outputs=decision_plot,
        )
        
    with gr.Tab("Documentation & Parameters"):
        gr.HTML("""
            <div style="text-align: center;">
            <h2>Documentation and Parameters</h2>
            <p>Here you can find the documentation and parameters for the estimation of safe navigation speed for autonomous vehicles.</p>
            </div>
        """)

demo.launch(share=True)