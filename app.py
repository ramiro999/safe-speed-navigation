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
    return fig_detr

# Función para el cálculo de distancia (sin detección de objetos)
def calculate_distance(mu, t, l, B):
    _, distance_plot_path1, distance_plot_path2, txt_path = calculate_lookahead_distance(mu, t, l, B, image_path=None)
    return distance_plot_path1, distance_plot_path2, txt_path

# Interfaz de Gradio con dos secciones
with gr.Blocks() as demo:
    gr.Markdown("# Lookahead Distance Calculation with DETR")
    gr.Markdown("This application allows you to detect objects in an image and calculate lookahead distance based on vehicle parameters.")

    # Sección de Detección de Objetos
    with gr.Column():
        gr.Markdown("## Object Detection Section")
        gr.Markdown("Upload an image and detect objects.")
        input_image = gr.Image(type="filepath", label="Input Image")
        
        # Imagen con objetos detectados
        result_plot = gr.Plot(label="Object Detection Results")
        
        # Botón Submit para la detección de objetos
        detect_btn = gr.Button("Detect Objects")
        detect_btn.click(
            fn=object_detection,
            inputs=input_image,
            outputs=result_plot
        )

    # Separador visual entre secciones
    gr.Markdown("---")

    # Sección de Cálculo de Distancia
    with gr.Column():
        gr.Markdown("## Lookahead Distance Calculation Section")
        gr.Markdown("Adjust parameters to calculate the lookahead distance.")

        # Parámetros de entrada para el cálculo de distancia
        mu = gr.Slider(0.0, 1.0, value=0.3, step=0.01, label="Friction Coefficient (mu)")
        t = gr.Slider(0.0, 5.0, value=0.2, step=0.01, label="Perception Time (t) [s]")
        l = gr.Slider(0.0, 5.0, value=0.25, step=0.01, label="Latency (l) [s]")
        B = gr.Slider(0.0, 5.0, value=2.0, step=0.1, label="Buffer (B) [m]")

        # Parámetros de entrada para el calculo de AOV
        hc = gr.Slider(0.0, 5.0, value=2.0, step=0.1, label="Camera Height (hc) [m]")
        thetaSlope = gr.Slider(0.0, 30.0, value=15.0, step=1.0, label="Slope Angle (thetaSlope) [deg]")

        # Colocar las dos gráficas en la misma fila
        with gr.Row():
            distance_plot1 = gr.Image(type="filepath", label="Lookahead Distance Plot")
            distance_plot2 = gr.Image(type="filepath", label="Angle of View Plot")

        txt_output = gr.File(label="Download Lookahead Distance Data")

        # Botón Submit para el cálculo de distancia
        calculate_btn = gr.Button("Calculate Distance")
        calculate_btn.click(
            fn=calculate_distance,
            inputs=[mu, t, l, B],
            outputs=[distance_plot1, distance_plot2, txt_output]
        )

demo.launch()
