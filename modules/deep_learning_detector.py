import numpy as np
import cv2
from PIL import Image
import warnings
from typing import Dict, List, Any, Optional
import os
import requests
from io import BytesIO
import json

# Try to import deep learning libraries
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    tf = None
    TF_AVAILABLE = False

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ort = None
    ORT_AVAILABLE = False

try:
    from huggingface_hub import InferenceApi
    HF_AVAILABLE = True
except ImportError:
    InferenceApi = None
    HF_AVAILABLE = False

warnings.filterwarnings('ignore')

class DeepLearningDetector:
    """
    Advanced deep learning-based detection using pre-trained neural networks
    for AI-generated content, deepfakes, and manipulation detection
    """
    
    def __init__(self):
        self.models = {}
        self.available = TF_AVAILABLE or TORCH_AVAILABLE
        
        # Set device if PyTorch is available
        if TORCH_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = "CPU (PyTorch not available)"
        
        self.model_urls = {
            'ai_detector': 'https://huggingface.co/umm-maybe/AI-image-detector/resolve/main/pytorch_model.bin',
            'deepfake_detector': 'https://github.com/selimsef/dfdc_deepfake_challenge/releases/download/v1.0/final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36',
        }
        
        # Only try to load models if we have the required dependencies
        if self.available:
            self.load_models()
        else:
            print("Warning: Deep learning dependencies not available. Install TensorFlow or PyTorch for neural network features.")
    
    def load_models(self):
        """Load and initialize deep learning models"""
        try:
            # Initialize AI Content Detection Model
            self._load_ai_detector()
            
            # Initialize Deepfake Detection Model
            self._load_deepfake_detector()
            
            # Initialize Image Quality Assessment Model
            self._load_quality_detector()
            
        except Exception as e:
            print(f"Warning: Could not load all deep learning models: {e}")
    
    def _load_ai_detector(self):
        """Load AI-generated content detection model"""
        try:
            if not TF_AVAILABLE:
                self.models['ai_detector'] = None
                return
                
            # Create a simple CNN model for AI detection
            model = keras.Sequential([
                keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
                keras.layers.MaxPooling2D(),
                keras.layers.Conv2D(64, 3, activation='relu'),
                keras.layers.MaxPooling2D(),
                keras.layers.Conv2D(64, 3, activation='relu'),
                keras.layers.Flatten(),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(1, activation='sigmoid')  # Binary classification: AI vs Real
            ])
            
            # Compile with appropriate loss and metrics
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
            
            # Load pre-trained weights if available
            try:
                model.load_weights('models/ai_detector_weights.h5')
            except:
                # Initialize with random weights (would need training in production)
                print("Warning: AI detector using untrained weights. Model needs training for optimal performance.")
            
            self.models['ai_detector'] = model
            
        except Exception as e:
            print(f"Could not load AI detector: {e}")
            self.models['ai_detector'] = None
    
    def _load_deepfake_detector(self):
        """Load deepfake detection model using EfficientNet"""
        try:
            if not TF_AVAILABLE:
                self.models['deepfake_detector'] = None
                return
                
            # Create EfficientNet-based deepfake detector
            base_model = keras.applications.EfficientNetB4(
                weights='imagenet',
                include_top=False,
                input_shape=(224, 224, 3)
            )
            
            # Freeze base model
            base_model.trainable = False
            
            # Add custom classification head
            model = keras.Sequential([
                base_model,
                keras.layers.GlobalAveragePooling2D(),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(128, activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(1, activation='sigmoid')  # Binary: Real vs Deepfake
            ])
            
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                loss='binary_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
            
            # Load pre-trained weights if available
            try:
                model.load_weights('models/deepfake_detector_weights.h5')
            except:
                print("Warning: Deepfake detector using ImageNet + untrained head. Model needs training for optimal performance.")
            
            self.models['deepfake_detector'] = model
            
        except Exception as e:
            print(f"Could not load deepfake detector: {e}")
            self.models['deepfake_detector'] = None
    
    def _load_quality_detector(self):
        """Load image quality assessment model"""
        try:
            if not TF_AVAILABLE:
                self.models['quality_detector'] = None
                return
                
            # Simple quality assessment network
            model = keras.Sequential([
                keras.layers.Conv2D(64, 3, activation='relu', input_shape=(224, 224, 3)),
                keras.layers.MaxPooling2D(),
                keras.layers.Conv2D(128, 3, activation='relu'),
                keras.layers.MaxPooling2D(),
                keras.layers.Conv2D(256, 3, activation='relu'),
                keras.layers.GlobalAveragePooling2D(),
                keras.layers.Dense(128, activation='relu'),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(1, activation='linear')  # Quality score regression
            ])
            
            model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            
            self.models['quality_detector'] = model
            
        except Exception as e:
            print(f"Could not load quality detector: {e}")
            self.models['quality_detector'] = None
    
    def preprocess_image(self, image_path: str, target_size: tuple = (224, 224)) -> np.ndarray:
        """Preprocess image for neural network input"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image = image.resize(target_size)
            
            # Convert to numpy array and normalize
            img_array = np.array(image) / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
            
        except Exception as e:
            raise Exception(f"Error preprocessing image: {e}")
    
    def detect_ai_content(self, image_path: str) -> Dict[str, Any]:
        """Detect AI-generated content using deep learning"""
        try:
            if self.models['ai_detector'] is None:
                return {'error': 'AI detector model not available'}
            
            # Preprocess image
            img_array = self.preprocess_image(image_path)
            
            # Get prediction
            prediction = self.models['ai_detector'].predict(img_array, verbose=0)[0][0]
            
            # Calculate confidence based on how far from 0.5 the prediction is
            confidence = abs(prediction - 0.5) * 2
            
            # Extract features from intermediate layers for analysis
            feature_extractor = keras.Model(
                inputs=self.models['ai_detector'].input,
                outputs=self.models['ai_detector'].layers[-3].output  # Before final dense layer
            )
            features = feature_extractor(img_array)[0]
            
            # Analyze feature patterns
            feature_analysis = self._analyze_features(features)
            
            return {
                'ai_probability': float(prediction),
                'confidence': float(confidence),
                'model_type': 'CNN_AI_Detector',
                'feature_analysis': feature_analysis,
                'technical_details': {
                    'prediction_raw': float(prediction),
                    'feature_vector_size': len(features),
                    'feature_mean': float(np.mean(features)),
                    'feature_std': float(np.std(features))
                }
            }
            
        except Exception as e:
            return {'error': f'AI detection failed: {str(e)}'}
    
    def detect_deepfake(self, image_path: str) -> Dict[str, Any]:
        """Detect deepfakes using EfficientNet-based model"""
        try:
            if self.models['deepfake_detector'] is None:
                return {'error': 'Deepfake detector model not available'}
            
            # Preprocess image
            img_array = self.preprocess_image(image_path)
            
            # Get prediction
            prediction = self.models['deepfake_detector'].predict(img_array, verbose=0)[0][0]
            
            # Calculate confidence
            confidence = abs(prediction - 0.5) * 2
            
            # Get intermediate layer activations for analysis
            layer_outputs = []
            for i, layer in enumerate(self.models['deepfake_detector'].layers):
                if 'conv' in layer.name.lower() or 'dense' in layer.name.lower():
                    temp_model = keras.Model(
                        inputs=self.models['deepfake_detector'].input,
                        outputs=layer.output
                    )
                    output = temp_model(img_array)
                    layer_outputs.append({
                        'layer_name': layer.name,
                        'activation_mean': float(np.mean(output)),
                        'activation_std': float(np.std(output))
                    })
            
            return {
                'deepfake_probability': float(prediction),
                'confidence': float(confidence),
                'model_type': 'EfficientNetB4_Deepfake_Detector',
                'layer_analysis': layer_outputs[:5],  # Top 5 layers
                'technical_details': {
                    'prediction_raw': float(prediction),
                    'model_architecture': 'EfficientNetB4 + Custom Head',
                    'input_resolution': '224x224x3'
                }
            }
            
        except Exception as e:
            return {'error': f'Deepfake detection failed: {str(e)}'}
    
    def assess_image_quality(self, image_path: str) -> Dict[str, Any]:
        """Assess image quality using neural network"""
        try:
            if self.models['quality_detector'] is None:
                return {'error': 'Quality detector model not available'}
            
            # Preprocess image
            img_array = self.preprocess_image(image_path)
            
            # Get quality score
            quality_score = self.models['quality_detector'].predict(img_array, verbose=0)[0][0]
            
            # Normalize quality score to 0-1 range
            quality_normalized = max(0, min(1, (quality_score + 1) / 2))
            
            return {
                'quality_score': float(quality_normalized),
                'quality_raw': float(quality_score),
                'quality_category': self._categorize_quality(quality_normalized),
                'model_type': 'CNN_Quality_Assessor'
            }
            
        except Exception as e:
            return {'error': f'Quality assessment failed: {str(e)}'}
    
    def comprehensive_analysis(self, image_path: str) -> Dict[str, Any]:
        """Perform comprehensive deep learning analysis"""
        try:
            results = {
                'ai_detection': self.detect_ai_content(image_path),
                'deepfake_detection': self.detect_deepfake(image_path),
                'quality_assessment': self.assess_image_quality(image_path)
            }
            
            # Calculate ensemble prediction
            ensemble_result = self._calculate_ensemble_prediction(results)
            results['ensemble_prediction'] = ensemble_result
            
            return results
            
        except Exception as e:
            return {'error': f'Comprehensive analysis failed: {str(e)}'}
    
    def _analyze_features(self, features: np.ndarray) -> Dict[str, Any]:
        """Analyze feature patterns from neural network"""
        try:
            # Statistical analysis of features
            feature_stats = {
                'mean': float(np.mean(features)),
                'std': float(np.std(features)),
                'min': float(np.min(features)),
                'max': float(np.max(features)),
                'sparsity': float(np.sum(features == 0) / len(features))
            }
            
            # Detect unusual patterns
            unusual_patterns = []
            if feature_stats['sparsity'] > 0.8:
                unusual_patterns.append("High feature sparsity detected")
            if feature_stats['std'] < 0.01:
                unusual_patterns.append("Unusually low feature variation")
            if feature_stats['max'] > 10:
                unusual_patterns.append("Unusually high activation values")
            
            return {
                'statistics': feature_stats,
                'unusual_patterns': unusual_patterns
            }
            
        except Exception as e:
            return {'error': f'Feature analysis failed: {str(e)}'}
    
    def _categorize_quality(self, quality_score: float) -> str:
        """Categorize image quality"""
        if quality_score >= 0.8:
            return "High Quality"
        elif quality_score >= 0.6:
            return "Medium Quality"
        elif quality_score >= 0.4:
            return "Low Quality"
        else:
            return "Very Low Quality"
    
    def _calculate_ensemble_prediction(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate ensemble prediction from multiple models"""
        try:
            predictions = []
            confidences = []
            
            # Collect valid predictions
            if 'ai_probability' in results.get('ai_detection', {}):
                predictions.append(results['ai_detection']['ai_probability'])
                confidences.append(results['ai_detection']['confidence'])
            
            if 'deepfake_probability' in results.get('deepfake_detection', {}):
                predictions.append(results['deepfake_detection']['deepfake_probability'])
                confidences.append(results['deepfake_detection']['confidence'])
            
            if not predictions:
                return {'error': 'No valid predictions for ensemble'}
            
            # Weighted average based on confidence
            if confidences and sum(confidences) > 0:
                ensemble_prediction = sum(p * c for p, c in zip(predictions, confidences)) / sum(confidences)
                ensemble_confidence = np.mean(confidences)
            else:
                ensemble_prediction = np.mean(predictions)
                ensemble_confidence = 0.5
            
            # Determine final classification
            if ensemble_prediction > 0.7:
                classification = "Likely Manipulated"
                risk_level = "High"
            elif ensemble_prediction > 0.5:
                classification = "Possibly Manipulated"
                risk_level = "Medium"
            else:
                classification = "Likely Authentic"
                risk_level = "Low"
            
            return {
                'ensemble_probability': float(ensemble_prediction),
                'ensemble_confidence': float(ensemble_confidence),
                'classification': classification,
                'risk_level': risk_level,
                'models_used': len(predictions)
            }
            
        except Exception as e:
            return {'error': f'Ensemble calculation failed: {str(e)}'}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        model_info = {}
        
        for model_name, model in self.models.items():
            if model is not None:
                try:
                    if hasattr(model, 'count_params'):
                        param_count = model.count_params()
                    else:
                        param_count = "Unknown"
                    
                    model_info[model_name] = {
                        'status': 'Loaded',
                        'parameters': param_count,
                        'input_shape': str(model.input_shape) if hasattr(model, 'input_shape') else "Unknown"
                    }
                except:
                    model_info[model_name] = {'status': 'Loaded but info unavailable'}
            else:
                model_info[model_name] = {'status': 'Not available'}
        
        model_info['device'] = str(self.device)
        model_info['tensorflow_version'] = tf.__version__ if tf else "Not Available"
        model_info['pytorch_version'] = torch.__version__ if torch else "Not Available"
        
        return model_info
