import streamlit as st
import requests
import numpy as np
import cv2
from PIL import Image
import io
import time

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(
    page_title="Smart Agriculture AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------
# CUSTOM CSS
# ---------------------------------
st.markdown(
    """
    <style>

    .stApp {
        background: linear-gradient(to right, #f4f7f9, #eef2f3);
    }

    .main-title {
        font-size: 48px;
        font-weight: 800;
        color: #1B5E20;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #555;
        margin-bottom: 30px;
    }

    .card {
        background: white;
        padding: 20px;
        border-radius: 18px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
        text-align: center;
    }

    .metric-title {
        font-size: 16px;
        color: black;
    }

    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #1B5E20;
    }

    .footer {
        text-align: center;
        color: #777;
        margin-top: 40px;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------
# HEADER
# ---------------------------------
st.markdown(
    '<div class="main-title">🌿 Smart Agriculture AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Real-Time Deep Learning Segmentation System using FastAPI + Streamlit</div>',
    unsafe_allow_html=True
)

# ---------------------------------
# SIDEBAR
# ---------------------------------
st.sidebar.title("⚙️ Configuration")

opacity = st.sidebar.slider(
    "Overlay Opacity",
    min_value=0.1,
    max_value=1.0,
    value=0.4,
    step=0.1
)

show_overlay = st.sidebar.checkbox(
    "Enable Overlay",
    value=True
)

# ---------------------------------
# FILE UPLOAD TITLE IN BLACK
# ---------------------------------
st.markdown(
    "<h4 style='color:black;'>📤 Upload Agricultural Image</h4>",
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png"]
)

# ---------------------------------
# MAIN LOGIC
# ---------------------------------
if uploaded_file is not None:

    # -----------------------------
    # READ IMAGE
    # -----------------------------
    file_bytes = uploaded_file.getvalue()

    image = cv2.imdecode(
        np.frombuffer(file_bytes, np.uint8),
        cv2.IMREAD_COLOR
    )

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (256, 256))

    # -----------------------------
    # SEND TO API
    # -----------------------------
    files = {
        "file": file_bytes
    }

    with st.spinner("🧠 AI model is processing the image..."):

        start_time = time.time()

        response = requests.post(
            "http://127.0.0.1:8000/predict",
            files=files
        )

        end_time = time.time()

    # -----------------------------
    # GET MASK
    # -----------------------------
    mask = np.array(response.json()["mask"])

    # FIXED VISUALIZATION
    binary_mask = (mask * 255).astype(np.uint8)

    # -----------------------------
    # OVERLAY CREATION
    # -----------------------------
    overlay = image.copy()

    overlay[binary_mask > 127] = [255, 0, 0]

    blended = cv2.addWeighted(
        image,
        1 - opacity,
        overlay,
        opacity,
        0
    )

    # -----------------------------
    # DASHBOARD CARDS
    # -----------------------------
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f'''
            <div class="card">
                <div class="metric-title">Image Width</div>
                <div class="metric-value">{image.shape[1]}</div>
            </div>
            ''',
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f'''
            <div class="card">
                <div class="metric-title">Image Height</div>
                <div class="metric-value">{image.shape[0]}</div>
            </div>
            ''',
            unsafe_allow_html=True
        )

    with c3:
        segmented_pixels = int((binary_mask > 127).sum())

        st.markdown(
            f'''
            <div class="card">
                <div class="metric-title">Segmented Pixels</div>
                <div class="metric-value">{segmented_pixels}</div>
            </div>
            ''',
            unsafe_allow_html=True
        )

    with c4:
        inference_time = round(end_time - start_time, 3)

        st.markdown(
            f'''
            <div class="card">
                <div class="metric-title">Inference Time</div>
                <div class="metric-value">{inference_time}s</div>
            </div>
            ''',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # -----------------------------
    # IMAGE DISPLAY
    # -----------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            "<h3 style='color:black;'>Original Image</h3>",
            unsafe_allow_html=True
        )
        st.image(image, use_container_width=True)

    with col2:
        st.markdown(
            "<h3 style='color:black;'>Predicted Mask</h3>",
            unsafe_allow_html=True
        )
        st.image(binary_mask, use_container_width=True)

    with col3:
        st.markdown(
            "<h3 style='color:black;'>Overlay Result</h3>",
            unsafe_allow_html=True
        )

        if show_overlay:
            st.image(blended, use_container_width=True)
        else:
            st.image(image, use_container_width=True)

    # -----------------------------
    # DOWNLOAD BUTTON
    # -----------------------------
    result_image = Image.fromarray(blended)

    buffer = io.BytesIO()
    result_image.save(buffer, format="PNG")

    st.download_button(
        label="⬇️ Download Result",
        data=buffer.getvalue(),
        file_name="segmentation_result.png",
        mime="image/png"
    )

# ---------------------------------
# FOOTER
# ---------------------------------
st.markdown(
    """
    <div class="footer">
        🌱 Smart Agriculture AI Dashboard | Deep Learning Segmentation System
    </div>
    """,
    unsafe_allow_html=True
)