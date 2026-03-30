import streamlit as st
import numpy as np
from PIL import Image

# Page Config MUST BE FIRST
st.set_page_config(
    page_title="DR Vision | Diabetic Retinopathy Detector",
    page_icon="👁",
    layout="centered"
)

MODEL_PATH = "dr_model.tflite"

# LOAD MODEL
@st.cache_resource
def load_model():
    try:
        import tflite_runtime.interpreter as tflite
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Could not load model: {e}")
        return None

interpreter = load_model()

if interpreter is not None:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
else:
    st.stop()


# PREPROCESS IMAGE
def preprocess_image(image):
    img = image.convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img).astype(np.float32)
    img = img / 255.0
    return img

# GRADES
grade_info = {
    0: ("No DR", "OK", "Healthy eye. No signs detected.", "#06d6a0"),
    1: ("Mild DR", "MILD", "Minor changes observed.", "#ffd166"),
    2: ("Moderate DR", "MOD", "Moderate retinal damage.", "#f4a261"),
    3: ("Severe DR", "SEV", "Severe damage detected.", "#ef476f"),
    4: ("Proliferative DR", "URGENT", "Immediate attention required.", "#9d4edd"),
}
# HERO UI (unchanged)
st.markdown("""
<div style="text-align:center;">
    <h1>👁 DR Vision</h1>
    <p>Diabetic Retinopathy Detection using AI</p>
</div>
""", unsafe_allow_html=True)
# UPLOAD
uploaded_file = st.file_uploader(
    "Upload fundus image",
    type=["jpg", "jpeg", "png"]
)

# PREDICTION
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Run Analysis"):
        with st.spinner("Analyzing..."):

            img = preprocess_image(image)
            img_input = np.expand_dims(img, axis=0)

            # 🔥 FIXED PREDICTION FLOW
            interpreter.set_tensor(input_details[0]['index'], img_input)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])

            predicted_grade = int(np.argmax(prediction))
            confidence = float(prediction[0][predicted_grade] * 100)

            grade_name, badge, desc, color = grade_info[predicted_grade]

            st.success(f"Prediction: Grade {predicted_grade} - {grade_name}")
            st.write(f"Confidence: {confidence:.2f}%")
            st.write(desc)
