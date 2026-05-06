import json
from PIL import Image
from ollama import Image
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from load_clip import model, processor, device


def extract_embeddings(data):
    image_embeddings = []
    text_embeddings = []

    for item in data:
        # ---- Image embedding ----
        image = Image.open(item["image"]).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            img_emb = model.get_image_features(**inputs)
        
        image_embeddings.append(img_emb.cpu().numpy()[0])
        
        # ---- Text embedding ----
        text = item["question"] + " " + item["context"]
        inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            txt_emb = model.get_text_features(**inputs)
        
        text_embeddings.append(txt_emb.cpu().numpy()[0])

    image_embeddings = np.array(image_embeddings).astype("float32")
    text_embeddings = np.array(text_embeddings).astype("float32")

    return image_embeddings, text_embeddings