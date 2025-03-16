# model_loader.py
import torch
import requests
#from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor

# Load model directly
from transformers import AutoImageProcessor, AutoModelForObjectDetection

def load_rtdetr_model():
    #model = RTDetrV2ForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r18vd")
    #tokenizer = AutoTokenizer.from_pretrained("KingRam/rtdetr-v2-r50-kitti-finetune-2")
    image_processor = AutoImageProcessor.from_pretrained("toukapy/detr_finetuned_kitti_mots-bright")
    model = AutoModelForObjectDetection.from_pretrained("toukapy/detr_finetuned_kitti_mots-bright")
    model.eval()
    return model, image_processor

