import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="DR Vision | Diabetic Retinopathy Detector",
    page_icon="👁",
    layout="centered"
)

MODEL_PATH = "dr_model.tflite"

# ======================
# LOAD MODEL
# ======================
@st.cache_resource
def load_model():
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

interpreter = load_model()

if interpreter is None:
    st.stop()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ======================
# PREPROCESS
# ======================
def preprocess_image(image):
    img = image.convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img).astype(np.float32) / 255.0
    return img

# ======================
# LABELS
# ======================
grade_info = {
    0: ("No DR", "OK", "Healthy eye."),
    1: ("Mild DR", "MILD", "Minor changes observed."),
    2: ("Moderate DR", "MOD", "Moderate damage."),
    3: ("Severe DR", "SEV", "Severe damage detected."),
    4: ("Proliferative DR", "URGENT", "Immediate attention required."),
}

# ======================
# UI
# ======================
st.title("👁 DR Vision")
st.write("Diabetic Retinopathy Detection using AI")

uploaded_file = st.file_uploader("Upload fundus image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)

    if st.button("Run Analysis"):
        with st.spinner("Analyzing..."):

            img = preprocess_image(image)
            img_input = np.expand_dims(img, axis=0).astype(np.float32)

            interpreter.set_tensor(input_details[0]['index'], img_input)
            interpreter.invoke()

            prediction = interpreter.get_tensor(output_details[0]['index'])

            predicted_class = int(np.argmax(prediction))
            confidence = float(prediction[0][predicted_class] * 100)

            name, badge, desc = grade_info[predicted_class]

            st.success(name)
            st.write(f"Confidence: {confidence:.2f}%")
            st.write(desc)
