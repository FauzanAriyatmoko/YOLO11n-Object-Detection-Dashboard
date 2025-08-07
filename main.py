import cv2
import requests
import streamlit as st
from pathlib import Path
import sys
import numpy as np
from ultralytics import YOLO
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path setup
FILE = Path(__file__).resolve()
ROOT = FILE.parent
if ROOT not in sys.path:
    sys.path.append(str(ROOT))
ROOT = ROOT.relative_to(Path.cwd())

# Model Configurations
MODEL_DIR = ROOT / 'weights'
DETECTION_MODEL = MODEL_DIR / 'yolo11n.pt'

# WebRTC Configuration
def get_ice_servers():
    """Fetch ICE server credentials from Metered.live if available, else use public STUN."""
    # Use Metered.live only if credentials are present and not running on localhost
    try:
        domain = st.secrets["METERED_DOMAIN"]
        secret_key = st.secrets["METERED_SECRET_KEY"]
        # Check if running on localhost
        if "localhost" in st.experimental_get_query_params().get("host", [""])[0] or "127.0.0.1" in st.experimental_get_query_params().get("host", [""])[0]:
            raise KeyError  # Force fallback to public STUN on localhost
        url = f"https://{domain}/api/v1/turn/credentials?apiKey={secret_key}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception:
        # Fallback for localhost or missing credentials
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": get_ice_servers()
})

@st.cache_resource
def load_yolo_model():
    """Load YOLO model once and cache it"""
    try:
        model_path = Path(DETECTION_MODEL)
        if not model_path.exists():
            st.error(f"Model file not found: {model_path}")
            st.error("Please make sure 'yolo11n.pt' is in the 'weights' folder")
            return None
        model = YOLO(str(model_path))
        st.success(f"✅ Model loaded successfully: {model_path.name}")
        return model
    except Exception as e:
        st.error(f"Unable to load model: {e}")
        return None

# Global variables for settings
if 'confidence' not in st.session_state:
    st.session_state.confidence = 0.4

# Page Layout
st.set_page_config(
    page_title="YOLO11 Webcam Detection",
    page_icon="🤖",
    layout="wide"
)

# Header
st.title("Real-time Object Detection with YOLO11📹")

# Load model
model = load_yolo_model()
if model is None:
    st.error("❌ Cannot proceed without a valid model. Please check your model file.")
    st.stop()

# Sidebar for settings
st.sidebar.header("⚙️ Detection Settings")
confidence_value = st.sidebar.slider(
    "🎚️ Confidence Threshold", 
    min_value=0.1, 
    max_value=1.0, 
    value=0.4, 
    step=0.05
)
st.sidebar.markdown("### 📹 Performance Settings")
frame_width = st.sidebar.selectbox("Frame Width", [320, 640, 1280], index=0)
frame_height = st.sidebar.selectbox("Frame Height", [240, 480, 720], index=0)

st.sidebar.markdown("### 📊 Model Information")
st.sidebar.info(f"""
**Model:** YOLO11n Detection  
**Classes:** {len(model.names)} objects  
**Confidence:** {confidence_value:.2f}
""")

def video_frame_callback(frame):
    """Process video frame for object detection"""
    try:
        img = frame.to_ndarray(format="bgr24")
        img_resized = cv2.resize(img, (frame_width, frame_height))
        results = model.predict(
            img_resized,
            conf=confidence_value,
            verbose=False,
            device='cpu'
        )
        if len(results) > 0:
            annotated_frame = results[0].plot()
            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")
        else:
            return av.VideoFrame.from_ndarray(img_resized, format="bgr24")
    except Exception as e:
        logger.error(f"Detection error: {e}")
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Main content
st.info("📝 Click **START** to begin real-time object detection using your webcam.")

webrtc_ctx = webrtc_streamer(
    key="yolo-webcam-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": {
            "width": {"min": 320, "ideal": frame_width, "max": 1280},
            "height": {"min": 240, "ideal": frame_height, "max": 720},
            "frameRate": {"min": 10, "ideal": 15, "max": 20}
        },
        "audio": False
    },
    async_processing=True,
    video_html_attrs={
        "style": {"width": "100%", "margin": "0 auto", "border": "2px solid #ff6b6b"},
        "controls": False,
        "autoPlay": True,
    }
)

# Sidebar Detection Status
st.sidebar.subheader("🎯 Detection Status")
if webrtc_ctx.state.playing:
    st.sidebar.success("🟢 **ACTIVE** - Detection in progress!")
    st.sidebar.markdown("**Current Settings:**")
    st.sidebar.write(f"📊 Confidence: {confidence_value:.2f}")
    st.sidebar.write(f"📐 Resolution: {frame_width}x{frame_height}")
    st.sidebar.write(f"🎯 Model: YOLO11n Detection")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**💡 Tips:**")
    st.sidebar.markdown("- Lower confidence = more detections")
    st.sidebar.markdown("- Lower resolution = better performance")
    st.sidebar.markdown("- Good lighting improves accuracy")
elif webrtc_ctx.state.signalling:
    st.sidebar.warning("🟡 **CONNECTING** - Setting up webcam...")
    st.sidebar.info("Please allow camera access when prompted")
else:
    st.sidebar.info("🔴 **INACTIVE** - Click START to begin")
    st.sidebar.markdown("**Ready to detect:**")
    st.sidebar.markdown("**🏷️ Detectable Objects:**")
    classes = list(model.names.values())
    cols = st.sidebar.columns(2)
    for i, cls in enumerate(classes[:20]):
        with cols[i % 2]:
            st.write(f"• {cls}")
    if len(classes) > 20:
        with st.sidebar.expander(f"View all {len(classes)} classes"):
            for cls in classes:
                st.write(f"• {cls}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 14px;'>"
    "🚀 Built with Streamlit, YOLOv11 & WebRTC | "
    "Real-time Object Detection Dashboard"
    "</div>",
    unsafe_allow_html=True
)
