 👁 DR Vision — Diabetic Retinopathy Detector
An AI-powered web app that detects Diabetic Retinopathy severity from fundus eye images using deep learning.
🔗 **Live App:** https://diabetic-retinopathy-detector-c85b4mqqv5mjeewouwkpxy.streamlit.app

🩺 About:
Diabetic Retinopathy is the leading cause of preventable blindness worldwide.
This tool classifies retinal fundus images into 5 severity grades instantly,
making early screening more accessible.

🔬 DR Severity Grades:
| Grade | Classification | Description |
|-------|---------------|-------------|
| 0 | No DR | Healthy eye |
| 1 | Mild | Minor changes |
| 2 | Moderate | Consult doctor soon |
| 3 | Severe | Urgent attention needed |
| 4 | Proliferative DR | Immediate intervention required |

🛠 Tech Stack:
- Language:Python 3.11
- Framework:Streamlit
- Model:TensorFlow Lite
- Dataset:APTOS 2019 Blindness Detection — Kaggle
- Deployment: Streamlit Cloud
- Image Processing: Pillow, NumPy

🚀 How to Run Locally:
bash
git clone https://github.com/Shanefatima12/diabetic-retinopathy-detector.git
cd diabetic-retinopathy-detector
pip install -r requirements.txt
streamlit run app.py

⚠️ Disclaimer:
This tool is for screening and educational purposes only.
It does not replace professional medical diagnosis.
Please consult a qualified ophthalmologist for clinical decisions.


👩‍💻 Built By:
Shane Fatima — Biomedical Engineer
Dataset: [APTOS 2019 on Kaggle](https://www.kaggle.com/competitions/aptos2019-blindness-detection)

