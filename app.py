import sys
import os
import torch
import gradio as gr
import cv2
#import uuid
#import hashlib

# Añadir directorios al path
sys.path.append('/home/ramiro-avila/simulation-gradio/stereo/NMRF')
sys.path.append('/home/ramiro-avila/simulation-gradio/stereo/NMRF/ops/setup/MultiScaleDeformableAttention')

# Importar módulos personalizados
from lookahead_calculator import calculate_lookahead_distance
from detr.image_processing import preprocess_image, plot_detr_results
from detr.model_loader import load_detr_model, COCO_INSTANCE_CATEGORY_NAMES
from stereo.NMRF.inference import run_inference
from stereo.NMRF.nmrf.data.datasets import KITTI
from stereo.NMRF.nmrf.utils.frame_utils import readDispVKITTI

# Cargar el modelo DETR
model = load_detr_model()

global_object_id = 1

# Función para solo la detección de objetos
def object_detection(image_path):
    image, image_tensor = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(image_tensor)

    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.8

    bboxes = outputs['pred_boxes'][0, keep].numpy()
    labels = probas[keep].argmax(-1).numpy()

    objects_info = []
    for idx, (bbox, label) in enumerate(zip(bboxes, labels), start=1):  # Usar un índice local
        cx, cy, w, h = bbox
        x0, y0 = int((cx - w / 2) * image.width), int((cy - h / 2) * image.height)
        x1, y1 = int((cx + w / 2) * image.width), int((cy + h / 2) * image.height)
        height_bb = abs(y1 - y0)

        objects_info.append({
            'id': idx,  # Usar el índice local como ID
            'class': COCO_INSTANCE_CATEGORY_NAMES[label],
            'height': height_bb,
            'bbox': [x0, y0, x1, y1]
        })

    # Extraer IDs y pasar a la función de renderizado
    ids = [obj['id'] for obj in objects_info]
    fig_detr = plot_detr_results(image, bboxes, labels, ids)
    fig_detr.savefig("object_detection_results.png", bbox_inches="tight")

    # Crear texto formateado con la información
    info_text = "Objetos detectados:\n\n"
    for obj in objects_info:
        info_text += f"ID: {obj['id']}\n"
        info_text += f"Clase: {obj['class']}\n"
        info_text += f"Áltura objeto: {obj['height']:,} píxeles\n"
        info_text += f"Coordenadas: ({obj['bbox'][0]}, {obj['bbox'][1]}) a ({obj['bbox'][2]}, {obj['bbox'][3]})\n\n"

    return "object_detection_results.png", info_text


# Modificación de la función `stereo_inference()`
def stereo_inference(image_path_left=None, image_path_right=None):
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
        # Ordenar los archivos por fecha de modificación para obtener el más reciente
        result_files.sort(key=os.path.getmtime, reverse=True)
        latest_result = result_files[0]  # Obtener el archivo más reciente
        latest_result_abs_path = os.path.abspath(latest_result)
        
        # Generar el mapa de profundidad a partir de la imagen de disparidad
        disp, valid = readDispVKITTI(latest_result_abs_path)

        # Guardar el mapa de profundidad como una imagen
        depth_map_filename = latest_result_abs_path.replace(".png", "_depth.png")
        cv2.imwrite(depth_map_filename, disp)

        # Devolver tanto el archivo de disparidad como el de profundidad
        return latest_result_abs_path, depth_map_filename
    else:
        raise FileNotFoundError("No se encontró ninguna imagen generada en la carpeta de resultados.")



# Función para el cálculo de distancia (sin detección de objetos)
def calculate_distance(mu, t, l, B):
    _, plot_path1, plot_path2, plot_path3, plot_path4 = calculate_lookahead_distance(mu, t, l, B, image_path=None)
    return plot_path1, plot_path2, plot_path3, plot_path4

# Interfaz de Gradio con dos secciones
with gr.Blocks() as demo:
    gr.Markdown("# Estimación de la velocidad de navegación segura para vehículos autónomos utilizando técnicas de visión por computadora")
    gr.Markdown("En este proyecto se propone desarrollar un simulador para estimar la velocidad de navegación segura para vehículos autónomos en términos de maniobras de detención y evasión de obstáculos detectados por medio de algoritmos de visión por computadora en imágenes RGB adquiridas por un sistema de visión estéreo.")

    # Sección de Detección de Objetos
    with gr.Column():
        gr.Markdown("## Sección de Detección de Objetos")
        input_image = gr.Image(type="filepath", label="Imagen de entrada")
        result_plot = gr.Image(type="filepath", label="Resultados de detección de objetos")
        result_info = gr.Textbox(label="Información de objetos detectados", lines=10)
        detect_btn = gr.Button("Detectar objetos")
        detect_btn.click(
            fn=object_detection,
            inputs=input_image,
            outputs=[result_plot, result_info]
        )

    # Separador visual entre secciones
    gr.Markdown("---")

    # Sección de Inferencia Estéreo
    with gr.Column():
        gr.Markdown("## Sección de Inferencia Estéreo")
        input_image_left = gr.Image(type="filepath", label="Imagen de entrada (izquierda)")
        input_image_right = gr.Image(type="filepath", label="Imagen de entrada (derecha)")
        result_plot_stereo = gr.Image(type="filepath", label="Mapa de disparidad")
        result_plot_depth = gr.Image(type="filepath", label="Mapa de profundidad")
        infer_btn = gr.Button("Realizar inferencia")
        infer_btn.click(
            fn=stereo_inference,
            inputs=[input_image_left, input_image_right],
            outputs=[result_plot_stereo, result_plot_depth]
        )



    # Separador visual entre secciones
    gr.Markdown("---")

    # Sección de Cálculo de Distancia
    with gr.Column():
        gr.Markdown("## Sección de calculo de la distancia segura")

        mu = gr.Slider(0.0, 1.0, value=0.3, step=0.01, label="Coeficiente de fricción (mu)")
        t = gr.Slider(0.0, 5.0, value=0.2, step=0.01, label="Tiempo de percepción (t) [s]")
        l = gr.Slider(0.0, 5.0, value=0.25, step=0.01, label="Latencia (l) [s]")
        B = gr.Slider(0.0, 5.0, value=2.0, step=0.1, label="Buffer (B) [m]")

        # Organizar las gráficas en dos filas
        with gr.Row():
            distance_plot1 = gr.Image(type="filepath", label="Gráfica de distancia segura")
            distance_plot2 = gr.Image(type="filepath", label="Gráfica del ángulo de visión")
        with gr.Row():
            distance_plot3 = gr.Image(type="filepath", label="Campo de visión Instantaneo Positivo [miliradianes]")
            distance_plot4 = gr.Image(type="filepath", label="Campo de visión instantaneo para los obstáculos")

        calculate_btn = gr.Button("Calcular distancia segura")
        calculate_btn.click(
            fn=calculate_distance,
            inputs=[mu, t, l, B],
            outputs=[distance_plot1, distance_plot2, distance_plot3, distance_plot4]
        )

demo.launch()
