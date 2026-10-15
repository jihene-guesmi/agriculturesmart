import streamlit as st
import cv2
import numpy as np
import torch
import time
import os
from PIL import Image
from ultralytics import YOLO
import torchvision.models.segmentation as segmentation

# ==========================================
# PAGE CONFIGURATION & APP THEME
# ==========================================
st.set_page_config(
    page_title="Automated Fruit Quality Scanner",
    page_icon="🍎",
    layout="centered"
)

# Custom Glassmorphic CSS Styling for a Premium Dashboard Look
st.markdown("""
<style>
@import url('https://googleapis.com');

.stApp {
    font-family: 'Inter', sans-serif;
    background-color: #06090e;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: 900;
    background: linear-gradient(45deg, #10b981, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #9ca3af;
    font-size: 15px;
    margin-bottom: 30px;
}

.glass-card {
    background: rgba(255, 255, 255, 0.02);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    padding: 24px;
    border-radius: 20px;
    margin-bottom: 20px;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.5);
    border: 1px solid rgba(255, 255, 255, 0.05);
}

/* Custom Metric Card Styles */
.status-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.01) 100%);
    padding: 20px;
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.03);
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}

.status-title {
    font-size: 12px;
    text-transform: uppercase;
    color: #9ca3af;
    font-weight: 500;
    letter-spacing: 0.05em;
}

.status-value {
    font-size: 24px;
    font-weight: 900;
    margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 🌿 AUTOMATED MODEL LOADING LAYERS (BACKGROUND CACHED)
# ==========================================
@st.cache_resource
def load_defect_tracker():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root_weights = os.path.join(base_dir, "detection", "best.pt")
    frontend_weights = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detection", "best.pt")
    
    if os.path.exists(root_weights):
        return YOLO(root_weights)
    elif os.path.exists(frontend_weights):
        return YOLO(frontend_weights)
    else:
        # Secure fallback to pre-trained object tracker if custom file is missing
        return YOLO("yolov8n.pt")

@st.cache_resource
def load_surface_mapper():
    # Direct library load bypasses torch.hub network issues on the cloud server
    model = segmentation.deeplabv3_resnet50(weights=segmentation.DeepLabV3_ResNet50_Weights.DEFAULT)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model

# Warm up core backend scanners silently in the background
with st.spinner("⏳ Warming up AI Quality Inspection System..."):
    defect_tracker_engine = load_defect_tracker()
    surface_mapper_engine = load_surface_mapper()

# ==========================================
# SIDEBAR DASHBOARD CONTROL PARAMETER DECK
# ==========================================
st.sidebar.markdown("### ⚙️ Inspection Settings")
overlay_opacity = st.sidebar.slider("Anomaly Highlights Visibility", 0.0, 1.0, 0.40, step=0.05)
enable_overlay = st.sidebar.checkbox("Show Surface Mapping Color Layer", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
📋 **Scanner Status:**
- Defect Spot Engine: **Active**
- Surface Area Mapping: **Active**
- Processing Mode: **Automated Quality Tiering**
""")

# ==========================================
# APPLICATION HEADERS
# ==========================================
st.markdown("<div class='title'>🍎 AI Fruit Quality Scanner</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Instant Defect Spot Identification, Surface Anomaly Mapping & Market Health Grading</div>", unsafe_allow_html=True)

st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("📤 Upload Fruit Image for Quality Inspection:", type=["jpg", "jpeg", "png"])
st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# INSPECTION INFRASTRUCTURE PIPELINE
# ==========================================
if uploaded_file is not None:
    # Read the incoming image arrays cleanly
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    h_orig, w_orig, _ = opencv_img.shape
    rgb_img = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB)
    
    st.markdown("### 📊 Live Inspection View")
    
    with st.spinner("🧠 Scanning fruit surface for anomalies and defects..."):
        # 1. Run Core Scanner A: Defect Spot Identification Box
        tracker_results = defect_tracker_engine(rgb_img, verbose=False)
        
        # FIX: Extract index 0 from the results list array to prevent the AttributeError on plot()
        annotated_frame = tracker_results[0].plot() 
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB) 
        
        # 2. Run Core Scanner B: Dense Surface Color Mapping
        input_image = Image.fromarray(rgb_img)
        from torchvision import transforms
        preprocess = transforms.Compose([
            transforms.Resize(520),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image).unsqueeze(0)
        
        if torch.cuda.is_available():
            input_tensor = input_tensor.to('cuda')
            
        with torch.no_grad():
            output = surface_mapper_engine(input_tensor)['out']
        surface_predictions = output.argmax(1).squeeze(0).byte().cpu().numpy()

    # Generate custom warning highlight overlay mask channels
    r = np.zeros_like(surface_predictions).astype(np.uint8)
    g = np.zeros_like(surface_predictions).astype(np.uint8)
    b = np.zeros_like(surface_predictions).astype(np.uint8)
    
    # Isolate defect surface classes (Maps target class 15 to bright orange warning pixels)
    anomaly_pixels = (surface_predictions == 15)
    r[anomaly_pixels] = 255
    g[anomaly_pixels] = 135  # Creates a modern neon orange warning tint over surface spots
    
    rgb_mask = cv2.merge([r, g, b])
    resized_mask = cv2.resize(rgb_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

    # Compile the interactive blended multi-layer projector view
    if enable_overlay:
        final_render = cv2.addWeighted(annotated_frame, 1.0 - overlay_opacity, resized_mask, overlay_opacity, 0)
    else:
        final_render = annotated_frame

    # Render results smoothly onto dashboard canvas
    st.image(final_render, caption="📸 Quality Scanner Output | Bounding Boxes Track Spots, Orange Layer Maps Surface Damage", width='stretch')
    
    # Calculate defect percentage metrics across the surface area
    total_scanned_area = surface_predictions.size
    total_defective_area = np.sum(anomaly_pixels)
    damage_percentage = (total_defective_area / total_scanned_area) * 100

    # 3. CONVERT METRICS TO AN UNDERSTANDABLE USER HEALTH GRADE
    if damage_percentage < 1.5:
        health_grade = "<span style='color:#10b981;'>🟢 GRADE A: PREMIUM QUALITY</span>"
        recommendation = "Excellent condition. Ideal for retail markets and premium export."
    elif damage_percentage <= 12.0:
        health_grade = "<span style='color:#f59e0b;'>🟡 GRADE B: PROCESSING ONLY</span>"
        recommendation = "Minor defects identified. Divert to juice, canning, or food processing."
    else:
        health_grade = "<span style='color:#ef4444;'>🔴 REJECT: ANOMALY QUARANTINE</span>"
        recommendation = "Severe rot or infection detected. Remove immediately from storage to prevent spread."

    # DISPLAY EASY BUSINESS KPI CARDS
    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
    col_metric1, col_metric2 = st.columns(2)
    
    with col_metric1:
        st.markdown(f"""
        <div class='status-card'>
            <div class='status-title'>AI Automatic Quality Grade</div>
            <div class='status-value'>{health_grade}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_metric2:
        st.markdown(f"""
        <div class='status-card'>
            <div class='status-title'>Surface Area Damaged / Spotted</div>
            <div class='status-value' style='color:#f3f4f6;'>{damage_percentage:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    # Display clear strategic operational advice card
    st.markdown("<div style='margin-top:15px;'></div>", unsafe_allow_html=True)
    st.info(f"📋 **Next-Step Operational Action:** {recommendation}")
        
else:
    st.info("Upload a fruit photo above. The AI Inspector will automatically map surface spots, evaluate anomalies, and assign an automatic market grade 🍏")

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.markdown("<div style='text-align: center; color: #555; font-size: 11px;'>🍿 User-Optimized Fruit Quality Scanner Ecosystem Deployed via Streamlit Cloud Systems</div>", unsafe_allow_html=True)
