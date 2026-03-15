import streamlit as st
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

class AgePredictorCNN(nn.Module):
    def __init__(self):
        super(AgePredictorCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AgePredictorCNN().to(device)
    model.load_state_dict(torch.load('age_prediction_model.pth', map_location=device))
    model.eval()
    return model, device

model, device = load_model()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

st.set_page_config(page_title="Age Prediction", layout="wide")
st.title("Age Prediction using Deep Learning")

img_file_buffer = st.camera_input("Take a picture to predict age")

if img_file_buffer is not None:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40))
    
    current_age = None
    
    for (x, y, w, h) in faces:
        padding = 40
        x_pad = max(0, x - padding)
        y_pad = max(0, y - int(padding * 1.5))
        w_pad = min(cv2_img.shape[1] - x_pad, w + (padding * 2))
        h_pad = min(cv2_img.shape[0] - y_pad, h + int(padding * 2.5))

        cv2.rectangle(cv2_img, (x_pad, y_pad), (x_pad + w_pad, y_pad + h_pad), (255, 0, 0), 2)
        face_img = cv2_img[y_pad:y_pad + h_pad, x_pad:x_pad + w_pad]
        
        if face_img.size > 0:
            try:
                face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(face_rgb)
                input_tensor = transform(pil_image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    current_age = output.item()
                    
                cv2.putText(cv2_img, f"Age: {current_age:.1f}", (x_pad, y_pad-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            except Exception as e:
                pass
                
    col_img, col_info = st.columns([1, 1])
    
    with col_img:
        st.image(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        
    with col_info:
        st.markdown("### 📊 Prediction")
        if current_age is not None:
            st.markdown(f"<h1 style='text-align: center; color: #4CAF50;'>{current_age:.1f} Years Old</h1>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='text-align: center;'>Cannot detect face</h3>", unsafe_allow_html=True)