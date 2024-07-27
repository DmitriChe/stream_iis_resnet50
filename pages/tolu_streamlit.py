import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from PIL import Image
import torch
from models.tolu_model import TolubaiResNet
from models.tolu_preprocessing import preprocess
import requests
from io import BytesIO
import time

idx2class = {0: 'Benign', 1: 'Malignant'}

@st.cache_resource()
def load_model():
    model = TolubaiResNet()
    model.load_state_dict(torch.load('models/tolu_model.pt', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

st.title('Skin cancer')

uploaded_files = st.file_uploader('Upload an image of a skin lesion', accept_multiple_files=True)
image_urls = st.text_area('Or enter image URLs (separate by commas)').split(',')

def predict(img):
    start_time = time.time()
    with torch.no_grad(): 
        output = model(img)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return output, elapsed_time

def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    return img

def display_prediction(image, caption):
    processed_image = preprocess(image)  
    output, elapsed_time = predict(processed_image)

    predicted_class = 1 if output.item() >= 0 else 0
    confidence = torch.sigmoid(output).item() if predicted_class == 1 else 1 - torch.sigmoid(output).item()

    st.image(image, caption=caption)
    st.write(f'Type: {idx2class[predicted_class]}, Probability: {confidence:.4f}')
    st.write(f'Model Response Time: {elapsed_time:.4f} seconds')
    
if uploaded_files:
    for file in uploaded_files:
        pil_image = Image.open(file).convert('RGB') 
        display_prediction(pil_image, caption=f'Uploaded Image: {file.name}')

if image_urls:
    for url in image_urls:
        url = url.strip()
        if url:
            try:
                pil_image = load_image_from_url(url)
                display_prediction(pil_image, caption=f'Image from URL: {url}')
            except Exception as e:
                st.error(f"Error loading image from URL {url}: {e}")
