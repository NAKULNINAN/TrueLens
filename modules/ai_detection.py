import cv2
import numpy as np
from PIL import Image
import os
from typing import Dict, List, Any
from scipy import ndimage, stats
from skimage import feature, measure, filters, segmentation
from skimage.feature import local_binary_pattern
# Note: greycomatrix and greycoprops were removed in newer scikit-image versions
# We'll implement simple alternatives or skip these features
from skimage.measure import shannon_entropy
import warnings
warnings.filterwarnings('ignore')

class AIContentDetector:
    """Detects AI-generated content using various heuristics and analysis"""
    
    def __init__(self):
        self.features = {
            'noise_analysis': True,
            'edge_analysis': True,
            'texture_analysis': True,
            'compression_artifacts': True,
            'statistical_analysis': True
        }
    
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """Analyze image for AI-generated content indicators"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise Exception("Could not load image")
            
            # Convert to different color spaces for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Perform various analyses
            results = {}
            features_detected = []
            
            # 1. Noise Analysis
            noise_score = self._analyze_noise_patterns(gray)
            results['noise_score'] = noise_score
            if noise_score > 0.7:
                features_detected.append("Unusual noise patterns")
            
            # 2. Edge Analysis
            edge_score = self._analyze_edge_consistency(gray)
            results['edge_score'] = edge_score
            if edge_score > 0.6:
                features_detected.append("Inconsistent edge patterns")
            
            # 3. Texture Analysis
            texture_score = self._analyze_texture_anomalies(gray)
            results['texture_score'] = texture_score
            if texture_score > 0.65:
                features_detected.append("Artificial texture patterns")
            
            # 4. Compression Artifacts
            compression_score = self._analyze_compression_artifacts(image)
            results['compression_score'] = compression_score
            if compression_score > 0.6:
                features_detected.append("Unusual compression artifacts")
            
            # 5. Statistical Analysis
            stats_score = self._analyze_statistical_properties(image)
            results['stats_score'] = stats_score
            if stats_score > 0.7:
                features_detected.append("Unusual statistical properties")
            
            # 6. Color Distribution Analysis
            color_score = self._analyze_color_distribution(hsv)
            results['color_score'] = color_score
            if color_score > 0.6:
                features_detected.append("Unnatural color distribution")
            
            # Calculate overall AI probability
            ai_probability = self._calculate_ai_probability(results)
            confidence = self._calculate_confidence(results, ai_probability)
            
            return {
                'ai_probability': ai_probability,
                'confidence': confidence,
                'features_detected': features_detected,
                'model_outputs': results,
                'analysis_details': {
                    'image_size': image.shape,
                    'channels': image.shape[2] if len(image.shape) == 3 else 1,
                    'mean_brightness': np.mean(gray),
                    'contrast_std': np.std(gray)
                }
            }
            
        except Exception as e:
            return {
                'ai_probability': 0.0,
                'confidence': 0.0,
                'features_detected': [],
                'model_outputs': {},
                'analysis_details': {},
                'error': str(e)
            }
    
    def _analyze_noise_patterns(self, gray_image: np.ndarray) -> float:
        """Advanced noise analysis with multiple sophisticated techniques"""
        try:
            h, w = gray_image.shape
            if h < 64 or w < 64:
                return 0.5  # Image too small for reliable analysis
            
            # 1. Multi-scale noise analysis
            scales = [1, 2, 4]
            noise_scores = []
            
            for scale in scales:
                kernel_size = 3 + 2 * scale
                blurred = cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), scale)
                noise = cv2.absdiff(gray_image.astype(np.float32), blurred.astype(np.float32))
                
                # Calculate noise characteristics
                noise_energy = np.sum(noise ** 2) / noise.size
                noise_entropy = shannon_entropy(noise.astype(np.uint8))
                
                # AI images often have very low noise entropy and energy
                if noise_entropy < 4.0 and noise_energy < 50:
                    noise_scores.append(0.9)  # Very suspicious
                elif noise_entropy < 5.5:
                    noise_scores.append(0.7)
                else:
                    noise_scores.append(0.2)
            
            # 2. Frequency domain noise analysis
            fft = np.fft.fft2(gray_image)
            fft_magnitude = np.abs(fft)
            
            # Check for unnatural frequency patterns
            high_freq_ratio = np.mean(fft_magnitude[h//4:, w//4:]) / (np.mean(fft_magnitude) + 1e-6)
            
            if high_freq_ratio < 0.1:  # Too little high frequency content
                noise_scores.append(0.8)
            elif high_freq_ratio > 0.5:
                noise_scores.append(0.3)
            else:
                noise_scores.append(0.1)
            
            # 3. Local Binary Pattern analysis for texture regularity
            radius = 3
            n_points = 8 * radius
            lbp = local_binary_pattern(gray_image, n_points, radius, method='uniform')
            
            # Calculate LBP histogram
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
            lbp_hist = lbp_hist.astype(float)
            lbp_hist /= (lbp_hist.sum() + 1e-6)
            
            # AI images often have too uniform LBP patterns
            lbp_entropy = -np.sum(lbp_hist * np.log(lbp_hist + 1e-10))
            max_entropy = np.log(len(lbp_hist))
            normalized_entropy = lbp_entropy / max_entropy
            
            if normalized_entropy < 0.6:  # Too uniform
                noise_scores.append(0.8)
            else:
                noise_scores.append(0.2)
            
            return np.mean(noise_scores)
            
        except Exception:
            return 0.0
    
    def _analyze_edge_consistency(self, gray_image: np.ndarray) -> float:
        """Analyze edge consistency and sharpness"""
        try:
            # Apply Canny edge detection
            edges = cv2.Canny(gray_image, 50, 150)
            
            # Calculate edge density and distribution
            edge_density = np.sum(edges > 0) / edges.size
            
            # Analyze edge gradient consistency
            grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # AI images sometimes have inconsistent gradients
            grad_std = np.std(gradient_magnitude)
            grad_mean = np.mean(gradient_magnitude)
            
            # Check for unnatural edge patterns
            if grad_mean > 0:
                inconsistency = grad_std / grad_mean
                
                # Look for specific AI artifacts
                # Very consistent edges (too perfect) or very inconsistent
                if inconsistency < 0.3:  # Too consistent, might be AI
                    score = 0.8
                elif inconsistency > 2.0:  # Too inconsistent
                    score = 0.9
                else:
                    score = min(1.0, inconsistency / 3.0)
            else:
                score = 0.5  # No edges detected, suspicious
            
            return score
            
        except Exception:
            return 0.0
    
    def _analyze_texture_anomalies(self, gray_image: np.ndarray) -> float:
        """Analyze texture patterns for AI-generated characteristics"""
        try:
            # Calculate Local Binary Pattern-like features
            rows, cols = gray_image.shape
            
            # Downsample for faster processing
            small_image = cv2.resize(gray_image, (min(200, cols), min(200, rows)))
            
            # Calculate texture energy using co-occurrence matrix approximation
            # Simple texture analysis using variance in local windows
            kernel_size = 5
            kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
            
            # Local mean
            local_mean = cv2.filter2D(small_image.astype(np.float32), -1, kernel)
            
            # Local variance (texture measure)
            local_var = cv2.filter2D((small_image.astype(np.float32) - local_mean)**2, -1, kernel)
            
            # AI-generated images often have specific texture characteristics
            texture_uniformity = np.std(local_var) / (np.mean(local_var) + 1e-6)
            
            # Normalize score
            score = min(1.0, texture_uniformity / 3.0)
            
            return score
            
        except Exception:
            return 0.0
    
    def _analyze_compression_artifacts(self, image: np.ndarray) -> float:
        """Analyze compression artifacts that might indicate AI generation"""
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply DCT to detect JPEG-like artifacts
            # Use small blocks similar to JPEG compression
            block_size = 8
            rows, cols = gray.shape
            
            artifact_scores = []
            
            for i in range(0, rows - block_size, block_size):
                for j in range(0, cols - block_size, block_size):
                    block = gray[i:i+block_size, j:j+block_size].astype(np.float32)
                    
                    # Simple artifact detection based on high frequency components
                    # Apply 2D FFT and analyze frequency distribution
                    fft_block = np.fft.fft2(block)
                    fft_magnitude = np.abs(fft_block)
                    
                    # Check for unusual high-frequency patterns
                    high_freq = fft_magnitude[block_size//2:, block_size//2:]
                    low_freq = fft_magnitude[:block_size//2, :block_size//2]
                    
                    if np.mean(low_freq) > 0:
                        ratio = np.mean(high_freq) / np.mean(low_freq)
                        artifact_scores.append(ratio)
            
            if artifact_scores:
                # AI images might have unusual frequency distributions
                avg_ratio = np.mean(artifact_scores)
                score = min(1.0, avg_ratio / 0.5)
            else:
                score = 0.0
            
            return score
            
        except Exception:
            return 0.0
    
    def _analyze_statistical_properties(self, image: np.ndarray) -> float:
        """Analyze statistical properties of the image"""
        try:
            # Convert to different color channels
            b, g, r = cv2.split(image)
            
            # Calculate various statistical measures
            channels = [b, g, r]
            
            # Benford's Law analysis (simplified)
            benford_scores = []
            for channel in channels:
                # Get first digits of pixel values
                non_zero = channel[channel > 0]
                if len(non_zero) > 0:
                    first_digits = []
                    for val in non_zero.flatten():
                        if val >= 10:
                            first_digit = int(str(val)[0])
                            first_digits.append(first_digit)
                    
                    if first_digits:
                        # Expected Benford distribution
                        expected_benford = [30.1, 17.6, 12.5, 9.7, 7.9, 6.7, 5.8, 5.1, 4.6]
                        
                        # Actual distribution
                        digit_counts = [first_digits.count(i+1) for i in range(9)]
                        total = sum(digit_counts)
                        
                        if total > 0:
                            actual_dist = [(count/total)*100 for count in digit_counts]
                            
                            # Calculate deviation from Benford's law
                            deviation = sum(abs(actual_dist[i] - expected_benford[i]) for i in range(9))
                            benford_scores.append(deviation / 100.0)
            
            if benford_scores:
                avg_benford_deviation = np.mean(benford_scores)
                score = min(1.0, avg_benford_deviation)
            else:
                score = 0.0
            
            return score
            
        except Exception:
            return 0.0
    
    def _analyze_color_distribution(self, hsv_image: np.ndarray) -> float:
        """Analyze color distribution for unnaturalness"""
        try:
            h, s, v = cv2.split(hsv_image)
            
            # Analyze hue distribution
            hue_hist = cv2.calcHist([h], [0], None, [180], [0, 180])
            hue_hist = hue_hist.flatten() / np.sum(hue_hist)
            
            # Analyze saturation distribution
            sat_hist = cv2.calcHist([s], [0], None, [256], [0, 256])
            sat_hist = sat_hist.flatten() / np.sum(sat_hist)
            
            # Check for unusual peaks or distributions
            # AI-generated images might have unnatural color distributions
            
            # Calculate entropy of distributions
            hue_entropy = -np.sum(hue_hist * np.log(hue_hist + 1e-10))
            sat_entropy = -np.sum(sat_hist * np.log(sat_hist + 1e-10))
            
            # Normalize entropies
            max_hue_entropy = np.log(180)
            max_sat_entropy = np.log(256)
            
            hue_norm = hue_entropy / max_hue_entropy
            sat_norm = sat_entropy / max_sat_entropy
            
            # Unusual distributions might indicate AI generation
            # Very high or very low entropy can be suspicious
            hue_score = abs(hue_norm - 0.7) / 0.7  # Deviation from expected natural entropy
            sat_score = abs(sat_norm - 0.6) / 0.6
            
            score = (hue_score + sat_score) / 2.0
            score = min(1.0, score)
            
            return score
            
        except Exception:
            return 0.0
    
    def _calculate_ai_probability(self, analysis_results: Dict) -> float:
        """Calculate overall AI generation probability"""
        # Updated weights for better detection
        weights = {
            'noise_score': 0.20,
            'edge_score': 0.25,
            'texture_score': 0.25,
            'compression_score': 0.10,
            'stats_score': 0.10,
            'color_score': 0.10
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for feature, weight in weights.items():
            if feature in analysis_results:
                weighted_sum += analysis_results[feature] * weight
                total_weight += weight
        
        if total_weight > 0:
            probability = weighted_sum / total_weight
        else:
            probability = 0.0
        
        return min(1.0, max(0.0, probability))
    
    def _calculate_confidence(self, analysis_results: Dict, ai_probability: float) -> float:
        """Calculate confidence in the AI detection result"""
        # Confidence based on consistency of different measures
        scores = [analysis_results.get(key, 0.0) for key in analysis_results.keys() if key.endswith('_score')]
        
        if len(scores) > 1:
            # Higher confidence when scores are consistent
            score_std = np.std(scores)
            score_mean = np.mean(scores)
            
            # Lower standard deviation = higher confidence
            consistency = 1.0 - min(1.0, score_std)
            
            # Confidence also depends on how extreme the probability is
            extremeness = abs(ai_probability - 0.5) * 2  # 0 to 1 scale
            
            confidence = (consistency + extremeness) / 2.0
        else:
            confidence = 0.5  # Medium confidence if we have limited data
        
        return min(1.0, max(0.0, confidence))
