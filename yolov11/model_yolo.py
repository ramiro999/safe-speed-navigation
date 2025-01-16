from ultralytics import YOLO

def load_yolov11_model(model_path="yolo11s.pt"):
    """
    Carga el modelo YOLOv11 desde un archivo de pesos.

    Args:
        model_path (str): Ruta del archivo de pesos del modelo.

    Returns:
        YOLO: Modelo YOLO cargado.
    """
    model = YOLO(model_path)  # Cargar el modelo YOLO
    #model.train(data='coco8.yaml', epochs=10)  # Entrenar el modelo con COCO
    return model
