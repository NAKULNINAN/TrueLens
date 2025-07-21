import os
from typing import Dict, Any
import mimetypes

def get_file_info(uploaded_file) -> Dict[str, Any]:
    """Extract file information from uploaded file"""
    file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # Size in MB
    file_name = uploaded_file.name
    
    # Determine file type and category
    mime_type, _ = mimetypes.guess_type(file_name)
    file_extension = os.path.splitext(file_name)[1].lower()
    
    # Categorize file type
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    
    if file_extension in image_extensions:
        category = 'image'
    elif file_extension in video_extensions:
        category = 'video'
    else:
        category = 'unknown'
    
    return {
        'name': file_name,
        'size': file_size,
        'type': mime_type or 'unknown',
        'extension': file_extension,
        'category': category
    }

def is_supported_format(uploaded_file) -> bool:
    """Check if the uploaded file format is supported"""
    file_info = get_file_info(uploaded_file)
    supported_categories = {'image', 'video'}
    return file_info['category'] in supported_categories

def format_confidence_score(confidence: float) -> str:
    """Format confidence score for display"""
    if confidence >= 0.9:
        return f"{confidence:.1%} (Very High)"
    elif confidence >= 0.7:
        return f"{confidence:.1%} (High)"
    elif confidence >= 0.5:
        return f"{confidence:.1%} (Medium)"
    elif confidence >= 0.3:
        return f"{confidence:.1%} (Low)"
    else:
        return f"{confidence:.1%} (Very Low)"

def get_risk_level_color(risk_level: str) -> str:
    """Get color code for risk level"""
    colors = {
        'Low': '#28a745',      # Green
        'Medium': '#ffc107',   # Yellow/Orange
        'High': '#dc3545',     # Red
        'Critical': '#6f42c1'  # Purple
    }
    return colors.get(risk_level, '#6c757d')  # Default gray

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    # Remove or replace unsafe characters
    unsafe_chars = '<>:"/\\|?*'
    for char in unsafe_chars:
        filename = filename.replace(char, '_')
    
    # Limit length
    name, ext = os.path.splitext(filename)
    if len(name) > 50:
        name = name[:50]
    
    return name + ext

def calculate_processing_time(start_time: float, end_time: float) -> str:
    """Calculate and format processing time"""
    duration = end_time - start_time
    
    if duration < 1:
        return f"{duration*1000:.0f}ms"
    elif duration < 60:
        return f"{duration:.1f}s"
    else:
        minutes = int(duration // 60)
        seconds = duration % 60
        return f"{minutes}m {seconds:.1f}s"

def validate_file_size(file_size_mb: float, max_size_mb: float = 100) -> tuple:
    """Validate file size and return (is_valid, message)"""
    if file_size_mb > max_size_mb:
        return False, f"File size ({file_size_mb:.1f}MB) exceeds maximum allowed size ({max_size_mb}MB)"
    elif file_size_mb < 0.01:  # Less than 10KB
        return False, "File size is too small to be a valid media file"
    else:
        return True, "File size is acceptable"

def get_supported_formats() -> Dict[str, list]:
    """Get list of supported file formats"""
    return {
        'images': ['JPEG', 'JPG', 'PNG', 'BMP', 'GIF', 'TIFF'],
        'videos': ['MP4', 'AVI', 'MOV', 'MKV', 'WMV', 'FLV', 'WEBM']
    }

def estimate_processing_time(file_info: Dict[str, Any]) -> str:
    """Estimate processing time based on file characteristics"""
    file_size = file_info['size']  # in MB
    category = file_info['category']
    
    if category == 'image':
        # Images are generally faster to process
        if file_size < 5:
            return "10-30 seconds"
        elif file_size < 20:
            return "30-60 seconds"
        else:
            return "1-2 minutes"
    
    elif category == 'video':
        # Videos take longer
        if file_size < 50:
            return "1-3 minutes"
        elif file_size < 200:
            return "3-8 minutes"
        else:
            return "8-15 minutes"
    
    else:
        return "Unknown"

def create_analysis_summary(results: Dict[str, Any]) -> str:
    """Create a human-readable summary of analysis results"""
    overall = results.get('overall_assessment', {})
    classification = overall.get('primary_classification', 'Unknown')
    confidence = overall.get('confidence', 0.0)
    risk_level = overall.get('risk_level', 'Unknown')
    
    summary_parts = [
        f"Classification: {classification}",
        f"Confidence: {confidence:.1%}",
        f"Risk Level: {risk_level}"
    ]
    
    # Add specific findings
    duplicate_result = results.get('duplicate_detection', {})
    if duplicate_result.get('is_duplicate', False):
        summary_parts.append("⚠️ Duplicate content detected")
    
    ai_result = results.get('ai_detection', {})
    if ai_result.get('ai_probability', 0.0) > 0.7:
        summary_parts.append("🤖 High AI generation probability")
    
    deepfake_result = results.get('deepfake_detection', {})
    if deepfake_result.get('deepfake_probability', 0.0) > 0.7:
        summary_parts.append("👤 High deepfake probability")
    
    return " | ".join(summary_parts)
