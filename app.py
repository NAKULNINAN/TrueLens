import streamlit as st
import os
import json
import tempfile
from PIL import Image
import cv2
import numpy as np
from modules.perceptual_hash import PerceptualHasher
from modules.ai_detection import AIContentDetector
from modules.deepfake_detection import DeepfakeDetector
from modules.utils import get_file_info, is_supported_format
from modules.video_processor import VideoProcessor

# Try to import deep learning detector
try:
    from modules.deep_learning_detector import DeepLearningDetector
    DEEP_LEARNING_AVAILABLE = True
except ImportError as e:
    print(f"Deep learning detector not available: {e}")
    DEEP_LEARNING_AVAILABLE = False
    DeepLearningDetector = None

# Page configuration
st.set_page_config(
    page_title="Media Authenticity Verification System",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'results' not in st.session_state:
    st.session_state.results = {}

# Initialize modules
@st.cache_resource
def load_models():
    """Load all detection models"""
    try:
        # Create necessary directories
        os.makedirs('data', exist_ok=True)
        
        # Initialize models with better error handling
        hasher = PerceptualHasher()
        ai_detector = AIContentDetector()
        deepfake_detector = DeepfakeDetector()
        video_processor = VideoProcessor()
        
        st.success("Models loaded successfully!")
        return hasher, ai_detector, deepfake_detector, video_processor
    except ImportError as e:
        st.error(f"Import error: {str(e)}. Please check if all required packages are installed.")
        st.info("Required packages: opencv-python, imagehash, pillow, numpy, scikit-image, moviepy")
        return None, None, None, None
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.info("This might be a temporary issue. Try refreshing the page.")
        return None, None, None, None

def main():
    st.title("🔍 Media Authenticity Verification System")
    st.markdown("Upload an image or video to verify its authenticity and detect duplicates, AI-generated content, or deepfakes.")
    
    # Load models
    hasher, ai_detector, deepfake_detector, video_processor = load_models()
    
    if None in [hasher, ai_detector, deepfake_detector, video_processor]:
        st.error("Failed to load detection models. Please check your installation.")
        return
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.7, 
            step=0.1,
            help="Minimum confidence score for positive detection"
        )
        
        show_technical_details = st.checkbox(
            "Show Technical Details", 
            value=False,
            help="Display hash values and model outputs"
        )
        
        use_deep_learning = st.checkbox(
            "🧠 Use Deep Learning (Neural Networks)", 
            value=DEEP_LEARNING_AVAILABLE,
            disabled=not DEEP_LEARNING_AVAILABLE,
            help="Enable advanced neural network analysis for higher accuracy"
        )
        
        if DEEP_LEARNING_AVAILABLE:
            st.success("🚀 Deep Learning: Available")
        else:
            st.warning("⚠️ Deep Learning: Not Available (install TensorFlow/PyTorch)")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov'],
        help="Supported formats: JPEG, PNG for images; MP4, AVI, MOV for videos"
    )
    
    if uploaded_file is not None:
        # Display file information
        file_info = get_file_info(uploaded_file)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📁 File Information")
            st.write(f"**Filename:** {file_info['name']}")
            st.write(f"**Size:** {file_info['size']:.2f} MB")
            st.write(f"**Type:** {file_info['type']}")
        
        with col2:
            if st.button("🔍 Analyze File", type="primary"):
                analyze_file(uploaded_file, hasher, ai_detector, deepfake_detector, video_processor, confidence_threshold, show_technical_details)
        
        # Display preview
        if file_info['category'] == 'image':
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
            except Exception as e:
                st.error(f"Error displaying image: {str(e)}")
        elif file_info['category'] == 'video':
            st.video(uploaded_file)
    
    # Display results if analysis is complete
    if st.session_state.analysis_complete and st.session_state.results:
        display_results(st.session_state.results, show_technical_details)

def analyze_file(uploaded_file, hasher, ai_detector, deepfake_detector, video_processor, confidence_threshold, show_technical_details):
    """Analyze uploaded file for authenticity"""
    
    # Reset previous results
    st.session_state.analysis_complete = False
    st.session_state.results = {}
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_file_path = tmp_file.name
        
        file_info = get_file_info(uploaded_file)
        results = {
            'file_info': file_info,
            'duplicate_detection': {},
            'ai_detection': {},
            'deepfake_detection': {},
            'overall_assessment': {}
        }
        
        # Step 1: Perceptual Hashing and Duplicate Detection
        status_text.text("🔄 Generating perceptual hash and checking for duplicates...")
        progress_bar.progress(20)
        
        if file_info['category'] == 'image':
            hash_result = hasher.analyze_image(temp_file_path)
        else:
            hash_result = video_processor.extract_frames_and_hash(temp_file_path, hasher)
        
        results['duplicate_detection'] = hash_result
        
        # Step 2: AI Content Detection
        status_text.text("🤖 Analyzing for AI-generated content...")
        progress_bar.progress(50)
        
        if file_info['category'] == 'image':
            ai_result = ai_detector.analyze_image(temp_file_path)
        else:
            ai_result = video_processor.analyze_video_for_ai(temp_file_path, ai_detector)
        
        results['ai_detection'] = ai_result
        
        # Step 3: Deepfake Detection
        status_text.text("👤 Performing deepfake detection...")
        progress_bar.progress(80)
        
        if file_info['category'] == 'image':
            deepfake_result = deepfake_detector.analyze_image(temp_file_path)
        else:
            deepfake_result = video_processor.analyze_video_for_deepfake(temp_file_path, deepfake_detector)
        
        results['deepfake_detection'] = deepfake_result
        
        # Step 4: Overall Assessment
        status_text.text("📊 Generating overall assessment...")
        progress_bar.progress(100)
        
        overall_assessment = generate_overall_assessment(results, confidence_threshold)
        results['overall_assessment'] = overall_assessment
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        # Store results and mark as complete
        st.session_state.results = results
        st.session_state.analysis_complete = True
        
        status_text.text("✅ Analysis complete!")
        progress_bar.empty()
        
        # Trigger rerun to display results
        st.rerun()
        
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        progress_bar.empty()
        status_text.empty()
        
        # Clean up temporary file if it exists
        try:
            os.unlink(temp_file_path)
        except:
            pass

def generate_overall_assessment(results, confidence_threshold):
    """Generate overall assessment based on all detection results"""
    
    assessment = {
        'primary_classification': 'Unknown',
        'confidence': 0.0,
        'risk_level': 'Low',
        'summary': '',
        'recommendations': []
    }
    
    # Extract key metrics
    is_duplicate = results['duplicate_detection'].get('is_duplicate', False)
    duplicate_confidence = results['duplicate_detection'].get('confidence', 0.0)
    
    ai_probability = results['ai_detection'].get('ai_probability', 0.0)
    ai_confidence = results['ai_detection'].get('confidence', 0.0)
    
    deepfake_probability = results['deepfake_detection'].get('deepfake_probability', 0.0)
    deepfake_confidence = results['deepfake_detection'].get('confidence', 0.0)
    
    # Improved classification logic with lower thresholds for better detection
    # Determine primary classification
    if is_duplicate and duplicate_confidence >= (confidence_threshold * 0.8):
        assessment['primary_classification'] = 'Duplicate'
        assessment['confidence'] = duplicate_confidence
        assessment['risk_level'] = 'Medium'
        assessment['summary'] = 'This content appears to be a duplicate of existing media.'
        assessment['recommendations'].append('Verify the source and originality of this content.')
    
    elif deepfake_probability >= (confidence_threshold * 0.6) and deepfake_confidence >= 0.5:
        assessment['primary_classification'] = 'Potential Deepfake'
        assessment['confidence'] = deepfake_confidence
        assessment['risk_level'] = 'High'
        assessment['summary'] = 'This content shows signs of being a deepfake or manipulated media.'
        assessment['recommendations'].extend([
            'Exercise extreme caution with this content.',
            'Verify through multiple independent sources.',
            'Consider professional forensic analysis.'
        ])
    
    elif ai_probability >= (confidence_threshold * 0.6) and ai_confidence >= 0.5:
        assessment['primary_classification'] = 'Potentially AI-Generated'
        assessment['confidence'] = ai_confidence
        assessment['risk_level'] = 'Medium'
        assessment['summary'] = 'This content appears to be generated by artificial intelligence.'
        assessment['recommendations'].extend([
            'Verify the source of this content.',
            'Be aware this may not represent real events or people.'
        ])
    
    elif ai_probability >= 0.4 or deepfake_probability >= 0.4:
        assessment['primary_classification'] = 'Suspicious'
        assessment['confidence'] = max(ai_confidence, deepfake_confidence)
        assessment['risk_level'] = 'Medium'
        assessment['summary'] = 'This content shows some indicators of artificial generation or manipulation.'
        assessment['recommendations'].extend([
            'Additional verification recommended.',
            'Some artificial characteristics detected.'
        ])
    
    else:
        assessment['primary_classification'] = 'Likely Original'
        assessment['confidence'] = max(0.7, max(1.0 - ai_probability, 1.0 - deepfake_probability))
        assessment['risk_level'] = 'Low'
        assessment['summary'] = 'This content appears to be original and authentic.'
        assessment['recommendations'].append('Content passed authenticity checks, but always verify sources when possible.')
    
    return assessment

def display_results(results, show_technical_details):
    """Display analysis results in a comprehensive dashboard"""
    
    st.subheader("📊 Analysis Results")
    
    # Overall Assessment Card
    overall = results['overall_assessment']
    
    # Color coding for risk levels
    risk_colors = {
        'Low': 'green',
        'Medium': 'orange',
        'High': 'red'
    }
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Classification",
            overall['primary_classification'],
            help="Primary classification based on analysis"
        )
    
    with col2:
        st.metric(
            "Confidence",
            f"{overall['confidence']:.2%}",
            help="Confidence in the classification"
        )
    
    with col3:
        risk_color = risk_colors.get(overall['risk_level'], 'gray')
        st.markdown(f"**Risk Level:** :{risk_color}[{overall['risk_level']}]")
    
    # Summary and Recommendations
    st.markdown(f"**Summary:** {overall['summary']}")
    
    if overall['recommendations']:
        st.markdown("**Recommendations:**")
        for rec in overall['recommendations']:
            st.markdown(f"• {rec}")
    
    # Detailed Results Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔄 Duplicate Detection", "🤖 AI Detection", "👤 Deepfake Detection", "🔧 Technical Details"])
    
    with tab1:
        dup_results = results['duplicate_detection']
        
        col1, col2 = st.columns(2)
        
        with col1:
            is_duplicate = dup_results.get('is_duplicate', False)
            st.markdown(f"**Duplicate Found:** {'Yes' if is_duplicate else 'No'}")
            st.markdown(f"**Confidence:** {dup_results.get('confidence', 0.0):.2%}")
        
        with col2:
            if dup_results.get('similar_hashes'):
                st.markdown(f"**Similar Items:** {len(dup_results['similar_hashes'])}")
                if show_technical_details:
                    st.markdown("**Similar Hash Values:**")
                    for hash_val in dup_results['similar_hashes'][:3]:
                        st.code(hash_val)
    
    with tab2:
        ai_results = results['ai_detection']
        
        col1, col2 = st.columns(2)
        
        with col1:
            ai_prob = ai_results.get('ai_probability', 0.0)
            st.markdown(f"**AI Probability:** {ai_prob:.2%}")
            st.markdown(f"**Confidence:** {ai_results.get('confidence', 0.0):.2%}")
        
        with col2:
            if ai_results.get('features_detected'):
                st.markdown("**AI Features Detected:**")
                for feature in ai_results['features_detected']:
                    st.markdown(f"• {feature}")
        
        if show_technical_details and ai_results.get('model_outputs'):
            st.markdown("**Model Outputs:**")
            st.json(ai_results['model_outputs'])
    
    with tab3:
        df_results = results['deepfake_detection']
        
        col1, col2 = st.columns(2)
        
        with col1:
            df_prob = df_results.get('deepfake_probability', 0.0)
            st.markdown(f"**Deepfake Probability:** {df_prob:.2%}")
            st.markdown(f"**Confidence:** {df_results.get('confidence', 0.0):.2%}")
        
        with col2:
            faces_detected = df_results.get('faces_detected', 0)
            st.markdown(f"**Faces Detected:** {faces_detected}")
            
            if df_results.get('anomalies_detected'):
                st.markdown("**Anomalies Detected:**")
                for anomaly in df_results['anomalies_detected']:
                    st.markdown(f"• {anomaly}")
        
        if show_technical_details and df_results.get('face_analysis'):
            st.markdown("**Face Analysis Details:**")
            st.json(df_results['face_analysis'])
    
    with tab4:
        if show_technical_details:
            st.markdown("**File Information:**")
            st.json(results['file_info'])
            
            st.markdown("**Perceptual Hash:**")
            st.code(results['duplicate_detection'].get('hash_value', 'N/A'))
            
            st.markdown("**Processing Metadata:**")
            processing_info = {
                'analysis_timestamp': results.get('timestamp', 'N/A'),
                'models_used': ['PerceptualHasher', 'AIContentDetector', 'DeepfakeDetector'],
                'processing_time': results.get('processing_time', 'N/A')
            }
            st.json(processing_info)
        else:
            st.info("Enable 'Show Technical Details' in the sidebar to view additional information.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application startup error: {str(e)}")
        st.info("Please refresh the page or contact support if the issue persists.")
        st.exception(e)
