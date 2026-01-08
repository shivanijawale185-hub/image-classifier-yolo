import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

st.title("YOLO Image Classification")

model = YOLO("yolo11n-cls.pt")

file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if file:
    img = Image.open(file)
    st.image(img)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        img.save(temp.name)
        result = model(temp.name)

    class_id = result[0].probs.top1
    st.write("Prediction:", result[0].names[class_id])
