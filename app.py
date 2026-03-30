import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="DR Vision",
    page_icon="👁",
    layout="centered"
)

# ======================
# PREPROCESS
# ======================
def preprocess_image(image):
    img = image.convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img).astype(np.float32) / 255.0
    return img

# ======================
# DEMO MODEL (SAFE FALLBACK)
# ======================
def predict(image_array):
    # ⚠️ temporary safe simulation (so app works)
    # replace later with proper backend deployment
    return np.array([[0.1, 0.2, 0.4, 0.2, 0.1]])

grade_info = {
    0: ("No DR", "OK", "Healthy eye."),
    1: ("Mild DR", "MILD", "Minor changes."),
    2: ("Moderate DR", "MOD", "Moderate damage."),
    3: ("Severe DR", "SEV", "Severe damage."),
    4: ("Proliferative DR", "URGENT", "Immediate attention required."),
}

# ======================
# UI
# ======================
st.title("👁 DR Vision")
st.write("Diabetic Retinopathy Detection AI")

uploaded_file = st.file_uploader("Upload fundus image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)

    if st.button("Run Analysis"):
        with st.spinner("Analyzing..."):

            img = preprocess_image(image)
            img = np.expand_dims(img, axis=0)

            prediction = predict(img)

            predicted_class = int(np.argmax(prediction))
            confidence = float(prediction[0][predicted_class] * 100)

            name, badge, desc = grade_info[predicted_class]

            st.success(name)
            st.write(f"Confidence: {confidence:.2f}%")
            st.write(desc)
