import streamlit as st
import cv2
import numpy as np
import torch
import time
import os
from PIL import Image
from ultralytics import YOLO

# ==========================================
# PAGE CONFIGURATION & THEME
# ==========================================
st.set_page_config(
    page_title="Smart Agriculture AI Studio",
    page_icon="🌿",
    layout="centered"
)

# Custom Glassmorphic CSS Styling Layouts
st.markdown("""
<style>
@import url('https://googleapis.com');

.stApp {
    font-family: 'Inter', sans-serif;
    background-color: #06090e;
}

.title {
    text-align: center;
    font-size: 46px;
    font-weight: 900;
    background: linear-gradient(45deg, #10b981, #059669);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #8e8e93;
    font-size: 15px;
    margin-bottom: 30px;
}

.glass-card {
    background: rgba(255, 255, 255, 0.02);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    padding: 22px;
    border-radius: 20px;
    margin-bottom: 20px;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.5);
    border: 1px solid rgba(255, 255, 255, 0.05);
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 🌿 CACHED MODEL INITIALIZATION (DIRECT ENGINE LOAD)
# ==========================================
@st.cache_resource
def load_yolo_model():
    # FIX: Explicit path resolution layer fixes the FileNotFoundError on Streamlit Cloud containers
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root_weights_path = os.path.join(base_dir, "detection", "best.pt")
    frontend_weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detection", "best.pt")
    local_fallback_path = "detection/best.pt"
    
    # Trace file presence across the directory framework variations
    if os.path.exists(root_weights_path):
        return YOLO(root_weights_path)
    elif os.path.exists(frontend_weights_path):
        return YOLO(frontend_weights_path)
    else:
        # Fallback tracking maps to direct cloud weight loading strings if files were too heavy for GitHub tracking
        return YOLO(local_fallback_path)

@st.cache_resource
def load_segmentation_model():
    # Instantiates the underlying DeepLabV3/ResNet-50 network securely
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model

# Warm up neural architectures directly into global server memory spaces
with st.spinner("⏳ Loading Concurrent Neural Acceleration Engines..."):
    yolo_engine = load_yolo_model()
    deeplab_engine = load_segmentation_model()

# ==========================================
# SIDEBAR RADAR PARAMETERS
# ==========================================
st.sidebar.markdown("### ⚙️ System Configuration")
overlay_opacity = st.sidebar.slider("Overlay Opacity Coefficient", 0.0, 1.0, 0.40, step=0.05)
enable_overlay = st.sidebar.checkbox("Enable Hybrid Processing Layer", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
📊 **Engine Telemetry Information:**
- Core Network A: **YOLO Backbone**
- Core Network B: **DeepLabV3/ResNet-50**
- Execution Device: **Cloud CPU Instance**
""")

# ==========================================
# APPLICATION BODY INTERFACE
# ==========================================
st.markdown("<div class='title'>🌿 Smart Agriculture AI Studio</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Real-Time Multimodal Deep Learning Parsing and Semantic Scene Segmentation</div>", unsafe_allow_html=True)

st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("📤 Upload Agronomic Spatial Crop Image:", type=["jpg", "jpeg", "png"])
st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# CORE LOCAL EVALUATION PIPELINE INTERACTION
# ==========================================
if uploaded_file is not None:
    # Read uploaded asset array maps
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    h_orig, w_orig, _ = opencv_img.shape
    rgb_img = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB)
    
    st.markdown("### 📊 Live Analytics Matrix View")
    
    with st.spinner("🧠 Computing Concurrent Dual-Engine Visual Inference Protocols..."):
        start_time = time.time()
        
        # --- Engine A Execution: Object Detection (YOLO) ---
        yolo_results = yolo_engine(rgb_img, verbose=False)
        annotated_frame = yolo_results[0].plot() # Automatically plots bounding boxes onto input tensor map
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB) # Keep mapping aligned
        
        # --- Engine B Execution: Dense Semantic Pixel Classification (DeepLabV3) ---
        input_image = Image.fromarray(rgb_img)
        preprocess = torch.hub.load('pytorch/vision:v0.10.0', 'transforms').Compose([
            torch.hub.load('pytorch/vision:v0.10.0', 'transforms').Resize(520),
            torch.hub.load('pytorch/vision:v0.10.0', 'transforms').ToTensor(),
            torch.hub.load('pytorch/vision:v0.10.0', 'transforms').Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ])
        input_tensor = preprocess(input_image).unsqueeze(0)
        
        if torch.cuda.is_available():
            input_tensor = input_tensor.to('cuda')
            
        with torch.no_grad():
            output = deeplab_engine(input_tensor)['out']
        seg_predictions = output.argmax(1).squeeze(0).byte().cpu().numpy()
        
        latency_bench = (time.time() - start_time) * 1000

    # Create distinct vegetative canvas mask channels
    r = np.zeros_like(seg_predictions).astype(np.uint8)
    g = np.zeros_like(seg_predictions).astype(np.uint8)
    b = np.zeros_like(seg_predictions).astype(np.uint8)
    
    # Isolate Pascal VOC class mapping 15 (Potted Plants/General Vegetation) 
    # to render an explicit green matrix layer
    plant_class_idx = (seg_predictions == 15)
    g[plant_class_idx] = 255 
    
    # Map residual secondary tracking layers to abstract pixels
    non_plant_idx = (seg_predictions != 15) & (seg_predictions != 0)
    r[non_plant_idx] = seg_predictions[non_plant_idx] * 12
    b[non_plant_idx] = 200 - (seg_predictions[non_plant_idx] * 8)
    
    rgb_mask = cv2.merge([r, g, b])
    # Resize spatial boundaries back onto original grid dimensions cleanly
    resized_mask = cv2.resize(rgb_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

    # Compile the alpha blended multi-layer projector overlays
    if enable_overlay:
        final_render = cv2.addWeighted(annotated_frame, 1.0 - overlay_opacity, resized_mask, overlay_opacity, 0)
    else:
        final_render = annotated_frame

    # Render results dynamically onto the client application workspace
    st.image(final_render, caption="📊 Parallel Multimodal Analysis Analytics Projector Overlay", width='stretch')
    
    # ==========================================
    # CORE METRICS DISPLAY FOOTER PANELS
    # ==========================================
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        # Compute agronomic crop canopy distribution area index ratio metrics
        canopy_density = (np.sum(plant_class_idx) / seg_predictions.size) * 100
        st.metric(label="🌿 Calculated Canopy Spatial Distribution", value=f"{canopy_density:.2f}%")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.metric(label="⚡ Execution Deterministic Benchmarking", value=f"{latency_bench:.1f} ms")
        st.markdown("</div>", unsafe_allow_html=True)
        
else:
    st.info("Upload an agricultural terrain tensor frame above to compute advanced spatial object detection overlays and pixel segmentation charts 🌾")

# ==========================================
# STUDIO RECONCILIATION FOOTER
# ==========================================
st.markdown("---")
st.markdown("<div style='text-align: center; color: #555; font-size: 11px;'>🍿 Architecture Engine Ecosystem Deployed via Streamlit Multi-Layer Vector Cloud Modules</div>", unsafe_allow_html=True)
