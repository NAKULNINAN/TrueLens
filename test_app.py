import streamlit as st
import sys
import os

st.set_page_config(
    page_title="Debug Test App",
    page_icon="🔧",
    layout="wide"
)

st.title("🔧 Debug Test App")

st.write("## System Information")
st.write(f"Python version: {sys.version}")
st.write(f"Current working directory: {os.getcwd()}")
st.write(f"Python path: {sys.path}")

st.write("## Environment Variables")
env_vars = dict(os.environ)
st.json(env_vars)

st.write("## Package Import Tests")

# Test basic imports
try:
    import numpy
    st.success("✅ numpy imported successfully")
    st.write(f"numpy version: {numpy.__version__}")
except Exception as e:
    st.error(f"❌ numpy import failed: {e}")

try:
    import cv2
    st.success("✅ opencv imported successfully")
    st.write(f"opencv version: {cv2.__version__}")
    
    # Test specific opencv functionality that might be causing issues
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        st.success("✅ opencv face cascade loaded successfully")
    except Exception as e:
        st.error(f"❌ opencv face cascade failed: {e}")
        
except Exception as e:
    st.error(f"❌ opencv import failed: {e}")

try:
    import imagehash
    st.success("✅ imagehash imported successfully")
except Exception as e:
    st.error(f"❌ imagehash import failed: {e}")

try:
    from PIL import Image
    st.success("✅ PIL imported successfully")
except Exception as e:
    st.error(f"❌ PIL import failed: {e}")

# Test custom modules
try:
    from modules.utils import get_file_info
    st.success("✅ modules.utils imported successfully")
except Exception as e:
    st.error(f"❌ modules.utils import failed: {e}")

try:
    from modules.perceptual_hash import PerceptualHasher
    st.success("✅ modules.perceptual_hash imported successfully")
except Exception as e:
    st.error(f"❌ modules.perceptual_hash import failed: {e}")

try:
    from modules.ai_detection import AIContentDetector
    st.success("✅ modules.ai_detection imported successfully")
except Exception as e:
    st.error(f"❌ modules.ai_detection import failed: {e}")

try:
    from modules.deepfake_detection import DeepfakeDetector
    st.success("✅ modules.deepfake_detection imported successfully")
except Exception as e:
    st.error(f"❌ modules.deepfake_detection import failed: {e}")

try:
    from modules.video_processor import VideoProcessor
    st.success("✅ modules.video_processor imported successfully")
except Exception as e:
    st.error(f"❌ modules.video_processor import failed: {e}")

st.write("## File System Check")
st.write("Checking if required directories exist:")

directories_to_check = ['data', 'modules', '.streamlit']
for directory in directories_to_check:
    if os.path.exists(directory):
        st.success(f"✅ Directory '{directory}' exists")
        if directory == 'modules':
            module_files = os.listdir(directory)
            st.write(f"Files in modules: {module_files}")
    else:
        st.error(f"❌ Directory '{directory}' missing")

# Test basic functionality
st.write("## Basic Functionality Test")

try:
    # Create test instance
    os.makedirs('data', exist_ok=True)
    hasher = PerceptualHasher()
    st.success("✅ PerceptualHasher instance created successfully")
except Exception as e:
    st.error(f"❌ PerceptualHasher instance creation failed: {e}")
    st.exception(e)

try:
    ai_detector = AIContentDetector()
    st.success("✅ AIContentDetector instance created successfully")
except Exception as e:
    st.error(f"❌ AIContentDetector instance creation failed: {e}")
    st.exception(e)

try:
    deepfake_detector = DeepfakeDetector()
    st.success("✅ DeepfakeDetector instance created successfully")
except Exception as e:
    st.error(f"❌ DeepfakeDetector instance creation failed: {e}")
    st.exception(e)

try:
    video_processor = VideoProcessor()
    st.success("✅ VideoProcessor instance created successfully")
except Exception as e:
    st.error(f"❌ VideoProcessor instance creation failed: {e}")
    st.exception(e)

st.write("## App Status")
st.success("🎉 Test app is running successfully!")
