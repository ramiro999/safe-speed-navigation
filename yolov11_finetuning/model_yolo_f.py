from ultralytics import YOLO

def load_yolov11_model_f(model_path="./yolov11_finetuning/yolov11n-kitti/train5/weights/last.pt"):
    """
    Carga el modelo YOLOv11 desde un archivo de pesos.

    Args:
        model_path (str): Ruta del archivo de pesos del modelo.

    Returns:
        YOLO: Modelo YOLO cargado.
    """
    model = YOLO(model_path)  # Cargar el modelo YOLO

    #train_results = model.train(
    #    data='../sample_data/kitti.yaml',
    #    epochs=25,  # Increase the number of epochs for better training
    #    patience=10,  # Increase patience to allow more epochs without improvement
    #    mixup=0.2,  # Increase mixup for better generalization
    #    project='yolov11n-kitti',
    #    classes=[0,1,2,3,4,5,6,7,8]  # Use only the classes of interest
    #)

    return model

# Cargar el modelo YOLOv11
#model = load_yolov11_model()