import streamlit as st
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

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
st.title("Real-time Age Prediction")

run_webcam = st.toggle("Open Webcam")

if run_webcam:
    col_video, col_info = st.columns([3, 1])
    
    with col_video:
        st.markdown("### 📷 Webcam")
        video_placeholder = st.empty()
        
    with col_info:
        st.markdown("### 📊 Prediction")
        age_placeholder = st.empty() 

    cap = cv2.VideoCapture(0)

    while run_webcam:
        ret, frame = cap.read()
        if not ret:
            st.error("Cannot read from webcam. Please check your camera.")
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40))

        current_age = None

        for (x, y, w, h) in faces:
            padding = 40
            x_pad = max(0, x - padding)
            y_pad = max(0, y - int(padding * 1.5))
            w_pad = min(frame.shape[1] - x_pad, w + (padding * 2))
            h_pad = min(frame.shape[0] - y_pad, h + int(padding * 2.5))

            cv2.rectangle(frame, (x_pad, y_pad), (x_pad + w_pad, y_pad + h_pad), (255, 0, 0), 2)
        
            face_img = frame[y_pad:y_pad + h_pad, x_pad:x_pad + w_pad]
            
            if face_img.size > 0:
                try:
                    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(face_rgb)
                    input_tensor = transform(pil_image).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        output = model(input_tensor)
                        current_age = output.item()
                        
                    cv2.putText(frame, f"Age: {current_age:.1f}", (x_pad, y_pad-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                except Exception as e:
                    pass
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB")
        
        if current_age is not None:
            age_placeholder.markdown(f"<h1 style='text-align: center; color: #4CAF50;'>{current_age:.1f} Years Old</h1>", unsafe_allow_html=True)
        else:
            age_placeholder.markdown("<h3 style='text-align: center;'>No face detected</h3>", unsafe_allow_html=True)

    cap.release()