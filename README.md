# Media Authenticity Verification System

A Python-based system that analyzes uploaded images and videos to detect duplicates, AI-generated content, and deepfakes using perceptual hashing and machine learning techniques.

## Features

- **Duplicate Detection**: Uses perceptual hashing to identify duplicate or similar content
- **AI Content Detection**: Analyzes noise patterns, edge consistency, and texture anomalies
- **Deepfake Detection**: Examines facial regions for manipulation indicators
- **Video Processing**: Frame-by-frame analysis for video files
- **Interactive Web Interface**: Streamlit-based dashboard with detailed results

## Installation

### Prerequisites
- Python 3.11+
- uv package manager (or pip)

### Setup
1. Clone or download this repository
2. Install dependencies:
   ```bash
   uv add opencv-python pillow imagehash moviepy numpy scikit-image streamlit
   ```
   Or with pip:
   ```bash
   pip install opencv-python pillow imagehash moviepy numpy scikit-image streamlit
   ```

### System Dependencies
For optimal performance, ensure you have:
- cmake (for advanced features)
- System libraries for image/video processing

## Usage

### Running the Application
```bash
streamlit run app.py --server.port 5000
```

### Web Interface
1. Open your browser to `http://localhost:5000`
2. Upload an image or video file
3. Click "Analyze File"
4. Review the comprehensive authenticity report

### Supported Formats
- **Images**: JPEG, PNG, BMP, GIF, TIFF
- **Videos**: MP4, AVI, MOV, MKV, WMV, FLV, WEBM

## Configuration

### Settings
- Adjust confidence thresholds in the sidebar
- Toggle technical details view
- Configure processing limits in module files

### Database
- Hash database stored in `data/hash_database.json`
- Automatically created on first run
- Stores perceptual hashes for duplicate detection

## Architecture

### Core Modules
- `modules/perceptual_hash.py` - Duplicate detection via hashing
- `modules/ai_detection.py` - AI-generated content analysis
- `modules/deepfake_detection.py` - Deepfake detection algorithms
- `modules/video_processor.py` - Video frame processing
- `modules/utils.py` - Utility functions

### Analysis Methods
1. **Perceptual Hashing**: Multiple hash types (average, perceptual, difference, wavelet)
2. **AI Detection**: Noise analysis, edge consistency, texture patterns, statistical properties
3. **Deepfake Detection**: Facial region analysis, texture smoothness, lighting consistency
4. **Video Analysis**: Frame sampling and temporal consistency checks

## Results Interpretation

### Classifications
- **Original**: Passes all authenticity checks
- **Duplicate**: Matches existing content in database
- **Potentially AI-Generated**: Shows AI generation indicators
- **Potential Deepfake**: Exhibits deepfake characteristics
- **Suspicious**: Some artificial indicators detected

### Confidence Scores
- **Very High (90%+)**: Strong evidence for classification
- **High (70-90%)**: Good confidence in result
- **Medium (50-70%)**: Moderate confidence
- **Low (30-50%)**: Weak indicators
- **Very Low (<30%)**: Minimal evidence

### Risk Levels
- **Low**: Content appears authentic
- **Medium**: Some concerns, verification recommended
- **High**: Strong indicators of manipulation

## Technical Details

### Performance Optimization
- Model caching for faster repeated analysis
- Frame sampling for efficient video processing
- Configurable processing limits

### Security Features
- Temporary file handling with automatic cleanup
- Input validation and format checking
- No permanent storage of user content

## Deployment

### Local Development
```bash
streamlit run app.py
```

### Production Deployment
1. Configure server settings in `.streamlit/config.toml`
2. Set up proper logging and monitoring
3. Consider database migration for large-scale use
4. Implement rate limiting and file size restrictions

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure all dependencies are installed
2. **Memory Issues**: Reduce video processing limits
3. **Performance**: Check system resources and file sizes

### Dependencies
If face-recognition installation fails:
- The system works without it using OpenCV-based alternatives
- Install cmake system dependency if needed

## License

This project is for educational and research purposes. Please ensure compliance with applicable laws and regulations when analyzing media content.

## Contributing

To extend the system:
1. Add new detection algorithms in the modules directory
2. Update weights and thresholds in analysis methods
3. Enhance the UI with additional features
4. Optimize performance for larger files

## Support

For technical issues:
1. Check the console logs for error details
2. Verify all dependencies are properly installed
3. Ensure input files are in supported formats
4. Review system resource availability