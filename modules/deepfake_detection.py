import cv2
import numpy as np
from typing import Dict, List, Any, Tuple
import os
from scipy import signal, ndimage
from scipy.spatial.distance import euclidean
from skimage import feature, measure, filters
from skimage.feature import local_binary_pattern, hog
from skimage.measure import shannon_entropy
import warnings
warnings.filterwarnings('ignore')

class DeepfakeDetector:
    """Detects deepfakes using facial analysis and inconsistency detection"""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """Analyze image for deepfake indicators"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise Exception("Could not load image")
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Find faces using Haar Cascade
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return {
                    'deepfake_probability': 0.0,
                    'confidence': 0.9,
                    'faces_detected': 0,
                    'anomalies_detected': [],
                    'face_analysis': {},
                    'message': 'No faces detected in image'
                }
            
            # Analyze each detected face
            face_analyses = []
            all_anomalies = []
            
            for i, (x, y, w, h) in enumerate(faces):
                face_location = (y, x+w, y+h, x)  # Convert to (top, right, bottom, left) format
                face_analysis = self._analyze_single_face(image, gray, face_location)
                face_analyses.append(face_analysis)
                all_anomalies.extend(face_analysis.get('anomalies', []))
            
            # Calculate overall deepfake probability
            deepfake_probability = self._calculate_deepfake_probability(face_analyses)
            confidence = self._calculate_confidence(face_analyses, deepfake_probability)
            
            return {
                'deepfake_probability': deepfake_probability,
                'confidence': confidence,
                'faces_detected': len(faces),
                'anomalies_detected': list(set(all_anomalies)),  # Remove duplicates
                'face_analysis': {
                    'individual_faces': face_analyses,
                    'average_scores': self._calculate_average_scores(face_analyses)
                }
            }
            
        except Exception as e:
            return {
                'deepfake_probability': 0.0,
                'confidence': 0.0,
                'faces_detected': 0,
                'anomalies_detected': [],
                'face_analysis': {},
                'error': str(e)
            }
    
    def _analyze_single_face(self, bgr_image: np.ndarray, gray_image: np.ndarray, 
                           face_location: Tuple) -> Dict[str, Any]:
        """Analyze a single face for deepfake indicators"""
        top, right, bottom, left = face_location
        
        # Extract face region
        face_bgr = bgr_image[top:bottom, left:right]
        face_gray = gray_image[top:bottom, left:right]
        
        analysis = {
            'location': face_location,
            'anomalies': [],
            'scores': {}
        }
        
        # 1. Facial region consistency (simplified without landmarks)
        region_score = self._analyze_facial_regions(face_gray)
        analysis['scores']['region_consistency'] = region_score
        if region_score > 0.6:
            analysis['anomalies'].append("Inconsistent facial regions")
        
        # 2. Eye analysis
        eye_score = self._analyze_eyes(face_bgr)
        analysis['scores']['eye_anomalies'] = eye_score
        if eye_score > 0.7:
            analysis['anomalies'].append("Eye region anomalies")
        
        # 3. Skin texture analysis
        texture_score = self._analyze_skin_texture(face_bgr)
        analysis['scores']['skin_texture'] = texture_score
        if texture_score > 0.65:
            analysis['anomalies'].append("Unnatural skin texture")
        
        # 4. Lighting consistency
        lighting_score = self._analyze_lighting_consistency(face_bgr)
        analysis['scores']['lighting_consistency'] = lighting_score
        if lighting_score > 0.6:
            analysis['anomalies'].append("Inconsistent lighting")
        
        # 5. Edge analysis around face
        edge_score = self._analyze_face_edges(bgr_image, face_location)
        analysis['scores']['edge_consistency'] = edge_score
        if edge_score > 0.7:
            analysis['anomalies'].append("Suspicious face boundaries")
        
        # 6. Color distribution analysis
        color_score = self._analyze_face_color_distribution(face_bgr)
        analysis['scores']['color_distribution'] = color_score
        if color_score > 0.6:
            analysis['anomalies'].append("Unnatural color distribution")
        
        # 7. Advanced frequency domain analysis
        freq_score = self._analyze_frequency_artifacts(face_gray)
        analysis['scores']['frequency_artifacts'] = freq_score
        if freq_score > 0.7:
            analysis['anomalies'].append("Suspicious frequency patterns")
        
        # 8. Micro-expression inconsistency analysis
        micro_score = self._analyze_micro_expressions(face_gray)
        analysis['scores']['micro_expressions'] = micro_score
        if micro_score > 0.6:
            analysis['anomalies'].append("Inconsistent micro-expressions")
        
        # 9. Symmetry analysis
        symmetry_score = self._analyze_facial_symmetry(face_gray)
        analysis['scores']['facial_symmetry'] = symmetry_score
        if symmetry_score > 0.7:
            analysis['anomalies'].append("Unnatural facial symmetry")
        
        return analysis
    
    def _analyze_facial_regions(self, face_gray: np.ndarray) -> float:
        """Analyze facial regions for consistency without landmarks"""
        try:
            h, w = face_gray.shape
            if h < 50 or w < 50:  # Face too small to analyze
                return 0.5
            
            # Divide face into regions and analyze consistency
            # Upper, middle, lower regions
            upper_region = face_gray[:h//3, :]
            middle_region = face_gray[h//3:2*h//3, :]
            lower_region = face_gray[2*h//3:, :]
            
            # Calculate texture consistency across regions
            upper_std = np.std(upper_region)
            middle_std = np.std(middle_region)
            lower_std = np.std(lower_region)
            
            # Check for unusual texture variations and patterns
            stds = [upper_std, middle_std, lower_std]
            means = [np.mean(upper_region), np.mean(middle_region), np.mean(lower_region)]
            
            std_variation = np.std(stds) / (np.mean(stds) + 1e-6)
            mean_variation = np.std(means) / (np.mean(means) + 1e-6)
            
            # Look for unnatural smoothness or inconsistency
            smoothness_score = 1.0 - min(1.0, np.mean(stds) / 30.0)  # Very smooth regions
            inconsistency_score = min(1.0, std_variation / 1.5)  # Inconsistent textures
            lighting_score = min(1.0, mean_variation / 0.3)  # Uneven lighting
            
            # Combine scores - deepfakes often have unnatural smoothness
            region_score = (smoothness_score * 0.5 + inconsistency_score * 0.3 + lighting_score * 0.2)
            
            return region_score
            
        except Exception:
            return 0.0
    

    
    def _analyze_eyes(self, face_bgr: np.ndarray) -> float:
        """Analyze eyes for deepfake indicators"""
        try:
            gray_face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
            
            # Detect eyes
            eyes = self.eye_cascade.detectMultiScale(gray_face, 1.1, 4)
            
            if len(eyes) < 2:
                return 0.3  # Suspicious if we can't detect both eyes clearly
            
            anomaly_scores = []
            
            for (ex, ey, ew, eh) in eyes:
                eye_region = gray_face[ey:ey+eh, ex:ex+ew]
                
                # Analyze eye region texture
                eye_std = np.std(eye_region)
                eye_mean = np.mean(eye_region)
                
                # Enhanced eye analysis for deepfake detection
                if eye_mean > 0:
                    texture_ratio = eye_std / eye_mean
                    
                    # Analyze eye region more thoroughly
                    eye_laplacian = cv2.Laplacian(eye_region, cv2.CV_64F).var()
                    
                    # Check for artificial smoothness (common in deepfakes)
                    if texture_ratio < 0.15:  # Very smooth eyes
                        anomaly_scores.append(0.9)
                    elif eye_laplacian < 50:  # Low detail variance
                        anomaly_scores.append(0.8)
                    elif texture_ratio > 1.0:  # Too noisy
                        anomaly_scores.append(0.6)
                    else:
                        anomaly_scores.append(0.1)
                else:
                    anomaly_scores.append(0.7)  # Dark eyes, suspicious
            
            return np.mean(anomaly_scores) if anomaly_scores else 0.0
            
        except Exception:
            return 0.0
    
    def _analyze_skin_texture(self, face_bgr: np.ndarray) -> float:
        """Analyze skin texture for unnaturalness"""
        try:
            # Convert to different color spaces for analysis
            gray_face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
            hsv_face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)
            
            # Analyze texture using Laplacian variance
            laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            
            # Analyze skin tone consistency
            h, s, v = cv2.split(hsv_face)
            
            # Calculate texture uniformity
            local_std = cv2.filter2D(gray_face.astype(np.float32), -1, np.ones((5,5))/25)
            texture_variation = np.std(local_std)
            
            # AI-generated faces might have overly smooth or inconsistent textures
            # Normalize scores
            smoothness_score = 1.0 - min(1.0, laplacian_var / 1000.0)  # Very smooth = suspicious
            variation_score = min(1.0, texture_variation / 50.0)  # Too much variation = suspicious
            
            # Combine scores
            texture_score = (smoothness_score + variation_score) / 2.0
            
            return texture_score
            
        except Exception:
            return 0.0
    
    def _analyze_lighting_consistency(self, face_bgr: np.ndarray) -> float:
        """Analyze lighting consistency across the face"""
        try:
            gray_face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
            
            # Divide face into regions
            h, w = gray_face.shape
            
            # Left, center, right regions
            left_region = gray_face[:, :w//3]
            center_region = gray_face[:, w//3:2*w//3]
            right_region = gray_face[:, 2*w//3:]
            
            # Calculate average brightness for each region
            left_brightness = np.mean(left_region)
            center_brightness = np.mean(center_region)
            right_brightness = np.mean(right_region)
            
            # Calculate lighting consistency
            brightness_values = [left_brightness, center_brightness, right_brightness]
            brightness_std = np.std(brightness_values)
            brightness_mean = np.mean(brightness_values)
            
            if brightness_mean > 0:
                inconsistency = brightness_std / brightness_mean
                # Normalize to 0-1 scale
                lighting_score = min(1.0, inconsistency * 3)
            else:
                lighting_score = 0.0
            
            return lighting_score
            
        except Exception:
            return 0.0
    
    def _analyze_face_edges(self, full_image: np.ndarray, face_location: Tuple) -> float:
        """Analyze edges around the face for blending artifacts"""
        try:
            top, right, bottom, left = face_location
            
            # Expand region slightly beyond face boundaries
            padding = 10
            expanded_top = max(0, top - padding)
            expanded_bottom = min(full_image.shape[0], bottom + padding)
            expanded_left = max(0, left - padding)
            expanded_right = min(full_image.shape[1], right + padding)
            
            region = full_image[expanded_top:expanded_bottom, expanded_left:expanded_right]
            gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            
            # Apply edge detection
            edges = cv2.Canny(gray_region, 50, 150)
            
            # Focus on the face boundary area
            face_height = bottom - top
            face_width = right - left
            
            # Define boundary region (around the face outline)
            boundary_mask = np.zeros_like(edges)
            
            # Create mask for face boundary
            center_y, center_x = face_height // 2 + padding, face_width // 2 + padding
            
            # Simple oval approximation for face boundary
            for y in range(edges.shape[0]):
                for x in range(edges.shape[1]):
                    # Distance from face center
                    dy = (y - center_y) / (face_height / 2)
                    dx = (x - center_x) / (face_width / 2)
                    
                    # Check if point is near face boundary
                    dist = np.sqrt(dx*dx + dy*dy)
                    if 0.8 <= dist <= 1.2:  # Near the face boundary
                        boundary_mask[y, x] = 1
            
            # Analyze edges in boundary region
            boundary_edges = edges * boundary_mask
            edge_density = np.sum(boundary_edges > 0) / np.sum(boundary_mask > 0) if np.sum(boundary_mask) > 0 else 0
            
            # Suspicious if there are too many or too few edges at the boundary
            if edge_density < 0.1 or edge_density > 0.4:
                edge_score = 0.8
            else:
                edge_score = 0.2
            
            return edge_score
            
        except Exception:
            return 0.0
    
    def _analyze_face_color_distribution(self, face_bgr: np.ndarray) -> float:
        """Analyze color distribution in the face"""
        try:
            # Convert to HSV for better color analysis
            face_hsv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(face_hsv)
            
            # Analyze skin tone distribution
            # Focus on areas that should be skin (exclude eyes, mouth, etc.)
            
            # Simple skin detection based on HSV ranges
            lower_skin = np.array([0, 20, 70])
            upper_skin = np.array([20, 255, 255])
            skin_mask = cv2.inRange(face_hsv, lower_skin, upper_skin)
            
            if np.sum(skin_mask) == 0:
                return 0.5  # No skin detected, medium suspicion
            
            # Analyze color consistency in skin regions
            skin_pixels_h = h[skin_mask > 0]
            skin_pixels_s = s[skin_mask > 0]
            skin_pixels_v = v[skin_mask > 0]
            
            # Calculate color distribution statistics
            h_std = np.std(skin_pixels_h)
            s_std = np.std(skin_pixels_s)
            v_std = np.std(skin_pixels_v)
            
            # AI-generated faces might have unusual color distributions
            # Very uniform or very varied colors can be suspicious
            
            # Normalize standard deviations
            h_score = min(1.0, h_std / 30.0)  # Hue should have some variation
            s_score = min(1.0, s_std / 50.0)  # Saturation variation
            v_score = min(1.0, v_std / 40.0)  # Value variation
            
            # Too little variation = suspicious (overly smooth)
            # Too much variation = suspicious (inconsistent)
            uniformity_scores = []
            for score in [h_score, s_score, v_score]:
                if score < 0.2:  # Too uniform
                    uniformity_scores.append(0.7)
                elif score > 0.8:  # Too varied
                    uniformity_scores.append(0.6)
                else:
                    uniformity_scores.append(0.2)
            
            color_score = np.mean(uniformity_scores)
            
            return color_score
            
        except Exception:
            return 0.0
    
    def _analyze_frequency_artifacts(self, face_gray: np.ndarray) -> float:
        """Advanced frequency domain analysis for deepfake artifacts"""
        try:
            h, w = face_gray.shape
            if h < 32 or w < 32:
                return 0.5
            
            # 1. FFT analysis for unnatural frequency patterns
            fft = np.fft.fft2(face_gray)
            fft_magnitude = np.abs(fft)
            fft_phase = np.angle(fft)
            
            # Analyze frequency distribution
            center_h, center_w = h // 2, w // 2
            
            # Low frequency components (DC and nearby)
            low_freq = fft_magnitude[:center_h//2, :center_w//2]
            # High frequency components
            high_freq = fft_magnitude[center_h//2:, center_w//2:]
            
            # Calculate frequency ratio
            low_energy = np.sum(low_freq ** 2)
            high_energy = np.sum(high_freq ** 2)
            
            if low_energy > 0:
                freq_ratio = high_energy / low_energy
                
                # Deepfakes often have unnatural frequency distributions
                if freq_ratio < 0.1:  # Too little high frequency
                    freq_score = 0.8
                elif freq_ratio > 2.0:  # Too much high frequency noise
                    freq_score = 0.7
                else:
                    freq_score = 0.3
            else:
                freq_score = 0.9  # Very suspicious
            
            # 2. Phase coherence analysis
            phase_coherence = np.std(fft_phase)
            if phase_coherence < 1.0:  # Too coherent, might be artificial
                freq_score = max(freq_score, 0.6)
            
            return freq_score
            
        except Exception:
            return 0.0
    
    def _analyze_micro_expressions(self, face_gray: np.ndarray) -> float:
        """Analyze micro-expressions and facial muscle consistency"""
        try:
            h, w = face_gray.shape
            if h < 64 or w < 64:
                return 0.5
            
            # Divide face into expression-relevant regions
            # Eyes region (upper 1/3)
            eyes_region = face_gray[:h//3, :]
            # Mouth region (lower 1/3)
            mouth_region = face_gray[2*h//3:, :]
            # Cheek regions (middle sides)
            left_cheek = face_gray[h//3:2*h//3, :w//3]
            right_cheek = face_gray[h//3:2*h//3, 2*w//3:]
            
            # Calculate gradient patterns in each region
            regions = [eyes_region, mouth_region, left_cheek, right_cheek]
            region_patterns = []
            
            for region in regions:
                if region.size > 100:  # Ensure minimum size
                    # Calculate HOG features for muscle pattern analysis
                    try:
                        hog_features = hog(region, orientations=8, pixels_per_cell=(8, 8),
                                         cells_per_block=(1, 1), visualize=False)
                        region_patterns.append(np.std(hog_features))
                    except:
                        region_patterns.append(0.0)
                else:
                    region_patterns.append(0.0)
            
            # Check for consistency between regions
            if len(region_patterns) > 1:
                pattern_std = np.std(region_patterns)
                pattern_mean = np.mean(region_patterns)
                
                if pattern_mean > 0:
                    inconsistency = pattern_std / pattern_mean
                    
                    # High inconsistency might indicate deepfake
                    if inconsistency > 1.5:
                        micro_score = 0.8
                    elif inconsistency < 0.2:  # Too consistent
                        micro_score = 0.7
                    else:
                        micro_score = 0.3
                else:
                    micro_score = 0.6
            else:
                micro_score = 0.5
            
            return micro_score
            
        except Exception:
            return 0.0
    
    def _analyze_facial_symmetry(self, face_gray: np.ndarray) -> float:
        """Analyze facial symmetry for unnatural patterns"""
        try:
            h, w = face_gray.shape
            if h < 50 or w < 50:
                return 0.5
            
            # Split face vertically into left and right halves
            mid_point = w // 2
            left_half = face_gray[:, :mid_point]
            right_half = face_gray[:, mid_point:]
            
            # Flip right half horizontally for comparison
            right_half_flipped = np.fliplr(right_half)
            
            # Ensure both halves have same dimensions
            min_width = min(left_half.shape[1], right_half_flipped.shape[1])
            left_half = left_half[:, :min_width]
            right_half_flipped = right_half_flipped[:, :min_width]
            
            # Calculate similarity between halves
            diff = np.abs(left_half.astype(np.float32) - right_half_flipped.astype(np.float32))
            mean_diff = np.mean(diff)
            
            # Calculate texture similarity
            left_std = np.std(left_half)
            right_std = np.std(right_half_flipped)
            
            texture_diff = abs(left_std - right_std) / (max(left_std, right_std) + 1e-6)
            
            # Real faces have natural asymmetry, perfect symmetry is suspicious
            if mean_diff < 5.0:  # Too similar
                symmetry_score = 0.9
            elif mean_diff > 40.0:  # Too different
                symmetry_score = 0.6
            elif texture_diff < 0.1:  # Texture too similar
                symmetry_score = 0.8
            else:
                symmetry_score = 0.2
            
            return symmetry_score
            
        except Exception:
            return 0.0
    
    def _calculate_deepfake_probability(self, face_analyses: List[Dict]) -> float:
        """Calculate overall deepfake probability from individual face analyses"""
        if not face_analyses:
            return 0.0
        
        # Weight different analysis components for better deepfake detection
        weights = {
            'region_consistency': 0.18,
            'eye_anomalies': 0.15,
            'skin_texture': 0.18,
            'lighting_consistency': 0.12,
            'edge_consistency': 0.08,
            'color_distribution': 0.05,
            'frequency_artifacts': 0.10,
            'micro_expressions': 0.08,
            'facial_symmetry': 0.06
        }
        
        all_probabilities = []
        
        for face_analysis in face_analyses:
            scores = face_analysis.get('scores', {})
            
            weighted_sum = 0.0
            total_weight = 0.0
            
            for component, weight in weights.items():
                if component in scores:
                    weighted_sum += scores[component] * weight
                    total_weight += weight
            
            if total_weight > 0:
                face_probability = weighted_sum / total_weight
                all_probabilities.append(face_probability)
        
        if all_probabilities:
            # Use the maximum probability (most suspicious face)
            return max(all_probabilities)
        else:
            return 0.0
    
    def _calculate_confidence(self, face_analyses: List[Dict], deepfake_probability: float) -> float:
        """Calculate confidence in the deepfake detection result"""
        if not face_analyses:
            return 0.0
        
        # Calculate consistency across different analysis methods
        all_scores = []
        for face_analysis in face_analyses:
            scores = face_analysis.get('scores', {})
            all_scores.extend(scores.values())
        
        if len(all_scores) > 1:
            score_std = np.std(all_scores)
            score_mean = np.mean(all_scores)
            
            # Higher consistency = higher confidence
            consistency = 1.0 - min(1.0, score_std)
            
            # Confidence also depends on how extreme the probability is
            extremeness = abs(deepfake_probability - 0.5) * 2
            
            confidence = (consistency + extremeness) / 2.0
        else:
            confidence = 0.5
        
        return min(1.0, max(0.0, confidence))
    
    def _calculate_average_scores(self, face_analyses: List[Dict]) -> Dict[str, float]:
        """Calculate average scores across all faces"""
        if not face_analyses:
            return {}
        
        all_score_types = set()
        for face_analysis in face_analyses:
            all_score_types.update(face_analysis.get('scores', {}).keys())
        
        average_scores = {}
        for score_type in all_score_types:
            scores = []
            for face_analysis in face_analyses:
                if score_type in face_analysis.get('scores', {}):
                    scores.append(face_analysis['scores'][score_type])
            
            if scores:
                average_scores[score_type] = np.mean(scores)
        
        return average_scores
