import streamlit as st
import numpy as np
from PIL import Image

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="DR Vision | Diabetic Retinopathy Detector",
    page_icon="👁",
    layout="centered"
)

# ======================
# HERO UI (RESTORED)
# ======================
st.markdown("""
<div style="text-align:center;">
    <h1>👁 DR Vision</h1>
    <p>Diabetic Retinopathy Detection using AI</p>
</div>
""", unsafe_allow_html=True)

# ======================
# PREPROCESS
# ======================
def preprocess_image(image):
    img = image.convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img).astype(np.float32) / 255.0
    return img

# ======================
# SAFE PREDICT (replace with your real model later)
# ======================
def predict(img):
    # If your model is already integrated elsewhere, replace this only
    return np.array([[0.05, 0.1, 0.6, 0.2, 0.05]])

# ======================
# LABELS
# ======================
grade_info = {
    0: ("No DR", "OK", "Healthy eye. No signs detected."),
    1: ("Mild DR", "MILD", "Minor changes observed."),
    2: ("Moderate DR", "MOD", "Moderate retinal damage."),
    3: ("Severe DR", "SEV", "Severe damage detected."),
    4: ("Proliferative DR", "URGENT", "Immediate attention required."),
}

# ======================
# UPLOAD SECTION
# ======================
uploaded_file = st.file_uploader(
    "Upload fundus image",
    type=["jpg", "jpeg", "png"]
)

# ======================
# DISPLAY + BUTTON
# ======================
if uploaded_file:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Run Analysis"):
        with st.spinner("Analyzing..."):

            img = preprocess_image(image)
            img = np.expand_dims(img, axis=0)

            prediction = predict(img)

            predicted_class = int(np.argmax(prediction))
            confidence = float(prediction[0][predicted_class] * 100)

            name, badge, desc = grade_info[predicted_class]

            st.success(f"Prediction: {name}")
            st.write(f"Confidence: {confidence:.2f}%")
            st.write(desc)
