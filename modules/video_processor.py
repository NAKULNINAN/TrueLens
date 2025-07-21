import cv2
import numpy as np
import os
import tempfile
from typing import Dict, List, Any

# Try to import moviepy, fall back gracefully if not available
try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    VideoFileClip = None

class VideoProcessor:
    """Handles video processing for authenticity verification"""
    
    def __init__(self):
        self.max_frames_to_analyze = 30  # Limit frames for performance
        self.frame_interval = 1.0  # Analyze every N seconds
    
    def extract_frames_and_hash(self, video_path: str, hasher) -> Dict[str, Any]:
        """Extract frames from video and perform perceptual hashing"""
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("Could not open video file")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            # Calculate frame sampling
            frame_interval = max(1, int(fps * self.frame_interval))
            frames_to_analyze = min(self.max_frames_to_analyze, total_frames // frame_interval)
            
            frame_hashes = []
            duplicate_frames = []
            
            for i in range(0, total_frames, frame_interval):
                if len(frame_hashes) >= frames_to_analyze:
                    break
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Save frame temporarily and hash it
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                    cv2.imwrite(tmp_file.name, frame)
                    
                    # Generate hash for this frame
                    frame_hash_result = hasher.analyze_image(tmp_file.name)
                    frame_hashes.append({
                        'frame_number': i,
                        'timestamp': i / fps if fps > 0 else 0,
                        'hash': frame_hash_result['hash_value'],
                        'is_duplicate': frame_hash_result['is_duplicate']
                    })
                    
                    if frame_hash_result['is_duplicate']:
                        duplicate_frames.append(i)
                    
                    # Clean up temporary file
                    os.unlink(tmp_file.name)
            
            cap.release()
            
            # Analyze overall video for duplicates
            duplicate_percentage = len(duplicate_frames) / len(frame_hashes) if frame_hashes else 0
            is_duplicate = duplicate_percentage > 0.3  # If 30% of frames are duplicates
            
            # Calculate confidence
            if is_duplicate:
                confidence = min(0.95, 0.5 + duplicate_percentage)
            else:
                confidence = max(0.8, 1.0 - duplicate_percentage * 2)
            
            return {
                'hash_value': frame_hashes[0]['hash'] if frame_hashes else '',
                'all_hashes': [fh['hash'] for fh in frame_hashes],
                'is_duplicate': is_duplicate,
                'confidence': confidence,
                'similar_hashes': [],  # Video-specific similar hashes would need separate implementation
                'similar_files': [],
                'video_analysis': {
                    'total_frames_analyzed': len(frame_hashes),
                    'duplicate_frames': len(duplicate_frames),
                    'duplicate_percentage': duplicate_percentage,
                    'video_duration': duration,
                    'fps': fps
                }
            }
            
        except Exception as e:
            return {
                'hash_value': '',
                'all_hashes': [],
                'is_duplicate': False,
                'confidence': 0.0,
                'similar_hashes': [],
                'similar_files': [],
                'video_analysis': {},
                'error': str(e)
            }
    
    def analyze_video_for_ai(self, video_path: str, ai_detector) -> Dict[str, Any]:
        """Analyze video frames for AI-generated content"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("Could not open video file")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Sample frames for analysis
            frame_interval = max(1, int(fps * self.frame_interval))
            frames_to_analyze = min(self.max_frames_to_analyze, total_frames // frame_interval)
            
            frame_results = []
            ai_probabilities = []
            
            for i in range(0, total_frames, frame_interval):
                if len(frame_results) >= frames_to_analyze:
                    break
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Save frame temporarily and analyze
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                    cv2.imwrite(tmp_file.name, frame)
                    
                    # Analyze frame for AI content
                    frame_ai_result = ai_detector.analyze_image(tmp_file.name)
                    
                    frame_results.append({
                        'frame_number': i,
                        'timestamp': i / fps if fps > 0 else 0,
                        'ai_probability': frame_ai_result['ai_probability'],
                        'confidence': frame_ai_result['confidence'],
                        'features_detected': frame_ai_result['features_detected']
                    })
                    
                    ai_probabilities.append(frame_ai_result['ai_probability'])
                    
                    # Clean up temporary file
                    os.unlink(tmp_file.name)
            
            cap.release()
            
            # Calculate overall video AI probability
            if ai_probabilities:
                # Use average and maximum for overall assessment
                avg_ai_probability = np.mean(ai_probabilities)
                max_ai_probability = np.max(ai_probabilities)
                
                # Weight toward higher values (more conservative)
                overall_ai_probability = (avg_ai_probability + max_ai_probability) / 2.0
                
                # Calculate confidence based on consistency
                prob_std = np.std(ai_probabilities)
                consistency = 1.0 - min(1.0, prob_std * 2)
                extremeness = abs(overall_ai_probability - 0.5) * 2
                confidence = (consistency + extremeness) / 2.0
                
                # Collect all unique features detected
                all_features = set()
                for frame_result in frame_results:
                    all_features.update(frame_result['features_detected'])
                
            else:
                overall_ai_probability = 0.0
                confidence = 0.0
                all_features = set()
            
            return {
                'ai_probability': overall_ai_probability,
                'confidence': confidence,
                'features_detected': list(all_features),
                'model_outputs': {
                    'frame_count': len(frame_results),
                    'average_probability': np.mean(ai_probabilities) if ai_probabilities else 0.0,
                    'max_probability': np.max(ai_probabilities) if ai_probabilities else 0.0,
                    'probability_std': np.std(ai_probabilities) if ai_probabilities else 0.0
                },
                'frame_analysis': frame_results[:5]  # Include details for first 5 frames
            }
            
        except Exception as e:
            return {
                'ai_probability': 0.0,
                'confidence': 0.0,
                'features_detected': [],
                'model_outputs': {},
                'frame_analysis': [],
                'error': str(e)
            }
    
    def analyze_video_for_deepfake(self, video_path: str, deepfake_detector) -> Dict[str, Any]:
        """Analyze video frames for deepfake content"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("Could not open video file")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Sample frames for analysis
            frame_interval = max(1, int(fps * self.frame_interval))
            frames_to_analyze = min(self.max_frames_to_analyze, total_frames // frame_interval)
            
            frame_results = []
            deepfake_probabilities = []
            total_faces_detected = 0
            all_anomalies = set()
            
            for i in range(0, total_frames, frame_interval):
                if len(frame_results) >= frames_to_analyze:
                    break
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Save frame temporarily and analyze
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                    cv2.imwrite(tmp_file.name, frame)
                    
                    # Analyze frame for deepfakes
                    frame_deepfake_result = deepfake_detector.analyze_image(tmp_file.name)
                    
                    frame_results.append({
                        'frame_number': i,
                        'timestamp': i / fps if fps > 0 else 0,
                        'deepfake_probability': frame_deepfake_result['deepfake_probability'],
                        'confidence': frame_deepfake_result['confidence'],
                        'faces_detected': frame_deepfake_result['faces_detected'],
                        'anomalies_detected': frame_deepfake_result['anomalies_detected']
                    })
                    
                    deepfake_probabilities.append(frame_deepfake_result['deepfake_probability'])
                    total_faces_detected += frame_deepfake_result['faces_detected']
                    all_anomalies.update(frame_deepfake_result['anomalies_detected'])
                    
                    # Clean up temporary file
                    os.unlink(tmp_file.name)
            
            cap.release()
            
            # Calculate overall video deepfake probability
            if deepfake_probabilities:
                # Use temporal consistency analysis
                overall_deepfake_probability = self._analyze_temporal_consistency(deepfake_probabilities)
                
                # Calculate confidence
                prob_std = np.std(deepfake_probabilities)
                consistency = 1.0 - min(1.0, prob_std * 2)
                extremeness = abs(overall_deepfake_probability - 0.5) * 2
                confidence = (consistency + extremeness) / 2.0
                
            else:
                overall_deepfake_probability = 0.0
                confidence = 0.0
            
            return {
                'deepfake_probability': overall_deepfake_probability,
                'confidence': confidence,
                'faces_detected': total_faces_detected,
                'anomalies_detected': list(all_anomalies),
                'face_analysis': {
                    'frames_analyzed': len(frame_results),
                    'frames_with_faces': sum(1 for fr in frame_results if fr['faces_detected'] > 0),
                    'average_faces_per_frame': total_faces_detected / len(frame_results) if frame_results else 0,
                    'temporal_consistency': self._calculate_temporal_consistency_score(deepfake_probabilities)
                },
                'frame_analysis': frame_results[:5]  # Include details for first 5 frames
            }
            
        except Exception as e:
            return {
                'deepfake_probability': 0.0,
                'confidence': 0.0,
                'faces_detected': 0,
                'anomalies_detected': [],
                'face_analysis': {},
                'frame_analysis': [],
                'error': str(e)
            }
    
    def _analyze_temporal_consistency(self, probabilities: List[float]) -> float:
        """Analyze temporal consistency of deepfake probabilities"""
        if len(probabilities) < 2:
            return np.mean(probabilities) if probabilities else 0.0
        
        # Calculate frame-to-frame changes
        changes = []
        for i in range(1, len(probabilities)):
            change = abs(probabilities[i] - probabilities[i-1])
            changes.append(change)
        
        # High temporal inconsistency might indicate deepfake
        avg_change = np.mean(changes)
        
        # Base probability from average
        base_prob = np.mean(probabilities)
        
        # Adjust based on temporal inconsistency
        inconsistency_bonus = min(0.3, avg_change * 2)  # Up to 30% bonus for inconsistency
        
        final_prob = min(1.0, base_prob + inconsistency_bonus)
        
        return final_prob
    
    def _calculate_temporal_consistency_score(self, probabilities: List[float]) -> float:
        """Calculate a score representing temporal consistency"""
        if len(probabilities) < 2:
            return 1.0  # Perfect consistency with single frame
        
        # Calculate variance in probabilities across frames
        prob_variance = np.var(probabilities)
        
        # Lower variance = higher consistency
        consistency_score = 1.0 - min(1.0, prob_variance * 4)
        
        return max(0.0, consistency_score)
    
    def extract_audio_features(self, video_path: str) -> Dict[str, Any]:
        """Extract audio features for additional analysis"""
        if not MOVIEPY_AVAILABLE:
            return {
                'has_audio': False,
                'error': 'MoviePy not available for audio processing'
            }
            
        try:
            video_clip = VideoFileClip(video_path)
            
            audio_info = {
                'has_audio': video_clip.audio is not None,
                'duration': video_clip.duration,
                'fps': video_clip.fps
            }
            
            if video_clip.audio:
                audio_info.update({
                    'audio_duration': video_clip.audio.duration,
                    'sample_rate': video_clip.audio.fps if hasattr(video_clip.audio, 'fps') else None
                })
            
            video_clip.close()
            
            return audio_info
            
        except Exception as e:
            return {
                'has_audio': False,
                'error': str(e)
            }
