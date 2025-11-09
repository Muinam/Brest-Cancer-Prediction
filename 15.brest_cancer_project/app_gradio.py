import gradio as gr
import joblib
import numpy as np

# -------------------- Load Trained Model --------------------
model = joblib.load(r"E:\Langchain_Model\15.brest_cancer_project\best_breast_cancer_model.pkl")

# -------------------- Prediction Function --------------------
def predict_breast_cancer(mean_radius, mean_texture, mean_smoothness, mean_compactness, mean_symmetry):
    features = np.array([[mean_radius, mean_texture, mean_smoothness, mean_compactness, mean_symmetry]])
    prediction = model.predict(features)[0]
    return "🟢 No Cancer Detected" if prediction == 0 else "🔴 Cancer Detected"

# -------------------- Custom CSS --------------------
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

body {
    background: radial-gradient(circle at top left, #0a0f1e 0%, #020617 60%, #0a192f 100%);
    font-family: 'Poppins', sans-serif;
    color: #e5e7eb;
    animation: fadeIn 1.2s ease-in-out;
}

/* Fade-in animation */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Glow background */
body::before {
    content: '';
    position: fixed;
    top: 50%;
    left: 50%;
    width: 600px;
    height: 600px;
    background: radial-gradient(circle, rgba(0,255,255,0.08) 0%, transparent 70%);
    transform: translate(-50%, -50%);
    filter: blur(60px);
    animation: pulse 6s infinite alternate;
}
@keyframes pulse {
    from { opacity: 0.3; transform: translate(-50%, -50%) scale(1); }
    to { opacity: 0.6; transform: translate(-50%, -50%) scale(1.1); }
}

/* Card container */
.gradio-container {
    background: rgba(15, 20, 35, 0.92);
    border-radius: 20px;
    padding: 30px 36px;
    box-shadow: 0px 0px 25px rgba(0, 255, 255, 0.05);
    border: 1px solid rgba(0, 255, 255, 0.12);
    backdrop-filter: blur(10px);
    max-width: 700px;
    margin: 50px auto !important;
}

/* Title Glow */
.title {
    text-align: center;
    font-size: 30px !important;
    color: #2ee6e6 !important;
    text-shadow: 0 0 20px rgba(46,230,230,0.75);
    animation: softglow 3s ease-in-out infinite alternate;
    margin-bottom: 6px;
}
@keyframes softglow {
    from { text-shadow: 0 0 10px #1bd4d4; }
    to { text-shadow: 0 0 25px #00ffff; }
}

/* Description */
.description {
    color: #c7e3f3 !important;
    font-size: 14px !important;
    text-align: center !important;
    margin-bottom: 18px;
}

/* Layout adjustments for inputs (rows) */
.gr-row {
    gap: 18px;
    margin-bottom: 12px;
}

/* Input fields (better contrast & visible values) */
input, .gr-number input {
    background-color: #0f172a !important;
    color: #ffffff !important;
    border-radius: 10px !important;
    border: 1px solid rgba(0, 255, 255, 0.20) !important;
    padding: 10px 12px !important;
    font-size: 15px !important;
    transition: all 0.22s ease;
    height: 44px !important;
}
.gr-number input::-webkit-outer-spin-button,
.gr-number input::-webkit-inner-spin-button {
    -webkit-appearance: none;
    margin: 0;
}
input::placeholder {
    color: rgba(200, 200, 200, 0.45) !important;
}
input:focus, .gr-number input:focus {
    border-color: #2ee6e6 !important;
    box-shadow: 0 0 10px rgba(46,230,230,0.55);
    outline: none !important;
}

/* Make labels a bit lighter and smaller */
label {
    color: #bfeaf0 !important;
    font-size: 13px !important;
    margin-bottom: 6px;
}

/* Predict Button */
.gr-button {
    background: linear-gradient(135deg, #00d4ff, #0099cc) !important;
    color: #ffffff !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    padding: 10px 18px !important;
    transition: all 0.25s ease;
}
.gr-button:hover {
    background: linear-gradient(135deg, #00ffff, #00bcd4) !important;
    transform: translateY(-2px) scale(1.02);
    box-shadow: 0 8px 20px rgba(0, 200, 255, 0.18);
}

/* Output box (keeps same style) */
.gr-textbox textarea {
    background: rgba(17, 25, 40, 0.62) !important;
    border: 1px solid rgba(0, 255, 255, 0.36) !important;
    color: #00ffff !important;
    font-weight: 600;
    border-radius: 10px;
    padding: 10px;
}

/* Small screens */
@media screen and (max-width: 768px) {
    .gradio-container {
        width: 92% !important;
        padding: 22px;
    }
    .title { font-size: 22px !important; }
}

/* Hide footer */
footer { visibility: hidden; }
"""

# -------------------- Gradio Blocks Layout (correct side-by-side rows) --------------------
with gr.Blocks(css=custom_css, theme=gr.themes.Default()) as demo:
    gr.Markdown("<div class='title'>🧠 Breast Cancer Prediction (AI Model)</div>")
    gr.Markdown("<div class='description'>Enter patient feature values to predict breast cancer using a trained ML model.</div>")

    with gr.Row():
        mean_radius = gr.Number(label="Mean Radius", placeholder="Enter mean radius")
        mean_texture = gr.Number(label="Mean Texture", placeholder="Enter mean texture")

    with gr.Row():
        mean_smoothness = gr.Number(label="Mean Smoothness", placeholder="Enter mean smoothness")
        mean_compactness = gr.Number(label="Mean Compactness", placeholder="Enter mean compactness")

    with gr.Row():
        mean_symmetry = gr.Number(label="Mean Symmetry", placeholder="Enter mean symmetry")

    predict_btn = gr.Button("Predict", elem_classes="gr-button")
    output = gr.Textbox(label="Prediction Result", interactive=False)

    predict_btn.click(
        fn=predict_breast_cancer,
        inputs=[mean_radius, mean_texture, mean_smoothness, mean_compactness, mean_symmetry],
        outputs=output
    )

# -------------------- Launch App --------------------
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=8501)
