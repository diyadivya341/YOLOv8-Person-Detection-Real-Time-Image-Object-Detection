# app.py
import streamlit as st
from ultralytics import YOLO
import os
from PIL import Image

st.title("🔍 YOLOv8 Object Detection on Images")

uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded image
    input_path = os.path.join("uploads", uploaded_file.name)
    os.makedirs("uploads", exist_ok=True)
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Load YOLO model
    model = YOLO("yolov8n.pt")  # Or your custom-trained model

    # Run detection
    results = model.predict(source=input_path, conf=0.5, save=True)
    output_path = results[0].save_dir

    st.success("✅ Detection complete!")
    
    # Display image with detections
    for file in os.listdir(output_path):
        if file.endswith(".jpg") or file.endswith(".png"):
            st.image(os.path.join(output_path, file), caption="Detected Image", use_column_width=True)
            break

