# pip3 install numpy --pre torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cpu

import streamlit as st
import torch
# import torchvision
# from torch.utils.data import DataLoader
# from torchvision import datasets
from torchvision import transforms as T

# Для чтения изображений с диска
from torchvision import io # input/output
# import torchutils as tu
import matplotlib.pyplot as plt
# import numpy as np
# import json
# import zipfile
# import os

import requests
from io import BytesIO
from PIL import Image

from models.model_dmche import MyResNet50


model = MyResNet50()
model.load_state_dict(torch.load('models/weights_dmche.pt', weights_only=False))

 
idx2class = {
    0: 'buildings',
    1: 'forest',
    2: 'glacier',
    3: 'mountain',
    4: 'sea',
    5: 'street'
    }


st.title("Распознавание датасета Intel Image Classification на дообученной ResNet50")

img_url = st.text_input("Введите url к картинке для распознавания")

st.write(img_url)
response = requests.get(img_url)
img = Image.open(BytesIO(response.content)).convert('RGB')

# Преобразования изображения с использованием torchvision
transform = T.Compose([
    T.Resize((150, 150)),  # Изменить размер изображения
    T.ToTensor()  # Преобразовать в тензор
])

img_tensor = transform(img)/255

torch.permute(img_tensor, (1, 2, 0))
with torch.inference_mode():
    preds = model(img_tensor.unsqueeze(0))
    pred_class = torch.argmax(preds, dim=1).item()
    class_name = idx2class[pred_class]


# Отображение картинки в Streamlit
st.text(class_name)
st.image(img, caption=class_name, use_column_width=True)
