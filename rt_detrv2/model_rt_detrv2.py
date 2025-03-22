# model_loader.py
import torch
import requests
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor

# Load model directly
from transformers import AutoImageProcessor, AutoModelForObjectDetection

def load_image_processor():
    image_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r101vd")
    return image_processor

def load_rtdetrv2_model():
    model = RTDetrV2ForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r101vd")
    model.eval()
    return model
