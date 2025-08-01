import streamlit as st
from PIL import Image
import numpy as np
import cv2
import google.generativeai as genai

# --- Page Setup ---
st.set_page_config(page_title="🥪 Sandwich Symmetry Evaluator", layout="centered")

# --- 🎨 Vibrant & Animated Custom CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

html, body, .stApp {
    background: linear-gradient(135deg, #FFDEE9, #B5FFFC);
    font-family: 'Poppins', sans-serif;
    color: #1a1a1a;
    animation: fadeIn 1s ease-in;
}

@keyframes fadeIn {
  from {opacity: 0;}
  to {opacity: 1;}
}

.stApp {
    padding: 2rem;
    border-radius: 20px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.1);
}

h1, h2, h3 {
    text-align: center;
    color: #2c003e;
    animation: popUp 0.6s ease-in-out;
}

@keyframes popUp {
  0% { transform: scale(0.9); opacity: 0; }
  100% { transform: scale(1); opacity: 1; }
}

.stButton>button {
    background: linear-gradient(135deg, #6e00ff, #c800ff);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.4rem;
    font-weight: bold;
    transition: 0.3s ease-in-out;
}

.stButton>button:hover {
    background: linear-gradient(135deg, #fc466b, #3f5efb);
    transform: scale(1.05);
}

hr {
    border: none;
    height: 2px;
    background: linear-gradient(to right, #f00, #0ff);
    margin: 2rem 0;
}

.score-badge {
    background: #111;
    color: #fff;
    padding: 1rem 2rem;
    margin: 1rem auto;
    border-radius: 20px;
    font-size: 1.5rem;
    font-weight: bold;
    text-align: center;
    box-shadow: 0 0 15px rgba(0,0,0,0.3);
    animation: glowPulse 1.8s infinite;
    width: fit-content;
}

@keyframes glowPulse {
  0% { box-shadow: 0 0 10px #ff00ff; }
  50% { box-shadow: 0 0 25px #00ffff; }
  100% { box-shadow: 0 0 10px #ff00ff; }
}
</style>
""", unsafe_allow_html=True)

# --- Gemini AI Setup ---
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

def generate_comment(score):
    prompt = f"""
You are a dramatic, funny, and snarky sandwich critic AI. A sandwich just scored {score}/100 in symmetry.
Make a short one-liner review that’s funny, clever, or brutally honest. Be original.
"""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"(AI is sulking: {e})"

# --- UI Elements ---
st.title("🥪 Sandwich Symmetry Evaluator")
st.subheader("Now with spice, sass, and spectacular gradients 🎉")
st.markdown("Upload your sandwich image. We'll tear it apart — visually, not literally 😎")

uploaded_file = st.file_uploader("Upload your sandwich masterpiece", type=["jpg", "jpeg", "png"])

def evaluate_symmetry(image_pil):
    image_pil = image_pil.resize((400, 400))
    gray = image_pil.convert("L")
    img = np.array(gray)

    h, w = img.shape
    mid = w // 2
    left = img[:, :mid]
    right = img[:, mid:]
    right_flipped = cv2.flip(right, 1)

    min_w = min(left.shape[1], right_flipped.shape[1])
    left = left[:, :min_w]
    right_flipped = right_flipped[:, :min_w]

    diff = cv2.absdiff(left, right_flipped)
    score = round(100 - (np.mean(diff) / 255 * 100), 2)

    heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    _, binary = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    return score, diff, heatmap, binary

# --- Logic ---
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Uploaded Sandwich", use_container_width=True)

    score, diff, heatmap, binary = evaluate_symmetry(image)

    st.markdown("<div class='score-badge'>📊 Symmetry Score: {:.2f} / 100</div>".format(score), unsafe_allow_html=True)

    with st.spinner("🤖 AI critic is judging..."):
        comment = generate_comment(score)

    st.success(f"💬 *{comment}*")

    st.markdown("---")
    st.subheader("🔍 Symmetry Visuals")
    