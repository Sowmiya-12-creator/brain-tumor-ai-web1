import streamlit as st
from PIL import Image
import numpy as np

from utils.unet_predict import unet_predict
from utils.yolo_predict import yolo_predict

st.title("🧠 Brain Tumor Detection AI")

uploaded = st.file_uploader("Upload MRI Image", type=["jpg", "png"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded MRI", use_column_width=True)

    tumor_present = yolo_predict(image)
    mask = unet_predict(image)

    if tumor_present:
        st.success("✅ Tumor Detected")
        st.image(mask, caption="Tumor Segmentation (U-Net)")
    else:
        st.warning("❌ No Tumor Detected")
