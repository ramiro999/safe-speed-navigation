from ultralytics import YOLO

def load_model(model_path):
    # Load a COCO-pretrained YOLO11n model
    model = YOLO(model_path)
    return model

def train_model(model, data_path, epochs, imgsz):
    # Train the model on the specified dataset
    results = model.train(data=data_path, epochs=epochs, imgsz=imgsz)
    return results

def run_inference(model, image_path):
    # Run inference with the YOLO11n model on the specified image
    results = model(image_path)
    return results

if __name__ == "__main__":
    model_path = "yolo11n.pt"
    data_path = "coco8.yaml"
    image_path = "path/to/bus.jpg"
    epochs = 100
    imgsz = 640

    model = load_model(model_path)
    train_results = train_model(model, data_path, epochs, imgsz)
    inference_results = run_inference(model, image_path)

    print(inference_results)