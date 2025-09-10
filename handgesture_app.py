import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

# Load trained model
model = load_model("hand_gesture_model.h5")

# Gesture classes (from Train/ folder)
gesture_classes = sorted(os.listdir(r"C:\Users\amrit\OneDrive\Desktop\SkillSoft4\Train"))

# Image size (must match training size)
img_size = (64, 64)

st.title("✋ Hand Gesture Recognition")
st.write("Upload a gesture image (from LeapGestRecog dataset or custom hand gesture)")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = load_img(uploaded_file, target_size=img_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(img_array)
    predicted_class = np.argmax(pred)
    confidence = np.max(pred)

    st.subheader("🔮 Prediction")
    st.write(f"Gesture: **{gesture_classes[predicted_class]}**")
    st.write(f"Confidence: **{confidence*100:.2f}%**")
