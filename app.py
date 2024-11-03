# app.py

import gradio as gr
from lookahead_calculator import calculate_lookahead_distance
from image_processing import preprocess_image, plot_detr_results
from model_loader import load_detr_model
import torch

# Cargar el modelo DETR
model = load_detr_model()

# Función para solo la detección de objetos
def object_detection(image_path):
    image, image_tensor = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(image_tensor)

    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.8

    bboxes = outputs['pred_boxes'][0, keep].numpy()
    labels = probas[keep].argmax(-1).numpy()

    fig_detr = plot_detr_results(image, bboxes, labels)
    fig_detr.savefig("object_detection_results.png", bbox_inches="tight")
    return "object_detection_results.png"

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
        detect_btn = gr.Button("Detectar objetos")
        detect_btn.click(
            fn=object_detection,
            inputs=input_image,
            outputs=result_plot
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
