from ultralytics import YOLO

def load_yolov11_model(model_path="yolo11n.pt"):
    """
    Carga el modelo YOLOv11 desde un archivo de pesos.

    Args:
        model_path (str): Ruta del archivo de pesos del modelo.

    Returns:
        YOLO: Modelo YOLO cargado.
    """
    model = YOLO(model_path)  # Cargar el modelo YOLO
    return model
