import imagehash
import json
import os
from PIL import Image
import numpy as np
from typing import Dict, List, Any
import cv2
from scipy.spatial.distance import hamming, euclidean
from skimage.feature import local_binary_pattern
from skimage.measure import structural_similarity as ssim
import warnings
warnings.filterwarnings('ignore')

class PerceptualHasher:
    """Handles perceptual hashing for duplicate detection"""
    
    def __init__(self, hash_db_path: str = "data/hash_database.json"):
        self.hash_db_path = hash_db_path
        self.hash_database = self._load_hash_database()
        self.similarity_threshold = 8  # Hamming distance threshold (more lenient for better detection)
    
    def _load_hash_database(self) -> Dict:
        """Load existing hash database or create new one"""
        try:
            if os.path.exists(self.hash_db_path):
                with open(self.hash_db_path, 'r') as f:
                    return json.load(f)
            else:
                return {"hashes": {}, "metadata": {"total_files": 0}}
        except Exception:
            return {"hashes": {}, "metadata": {"total_files": 0}}
    
    def _save_hash_database(self):
        """Save hash database to file"""
        try:
            os.makedirs(os.path.dirname(self.hash_db_path), exist_ok=True)
            with open(self.hash_db_path, 'w') as f:
                json.dump(self.hash_database, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save hash database: {e}")
    
    def generate_hashes(self, image_path: str) -> Dict[str, str]:
        """Generate multiple perceptual hashes for an image"""
        try:
            image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Generate different types of hashes
            hashes = {
                'average_hash': str(imagehash.average_hash(image)),
                'perceptual_hash': str(imagehash.phash(image)),
                'difference_hash': str(imagehash.dhash(image)),
                'wavelet_hash': str(imagehash.whash(image))
            }
            
            return hashes
            
        except Exception as e:
            raise Exception(f"Error generating hashes: {str(e)}")
    
    def find_similar_hashes(self, target_hashes: Dict[str, str]) -> List[Dict]:
        """Find similar hashes in the database"""
        similar_hashes = []
        
        for stored_filename, stored_hashes in self.hash_database["hashes"].items():
            similarities = {}
            
            for hash_type, target_hash in target_hashes.items():
                if hash_type in stored_hashes:
                    try:
                        # Calculate Hamming distance
                        target_hash_obj = imagehash.hex_to_hash(target_hash)
                        stored_hash_obj = imagehash.hex_to_hash(stored_hashes[hash_type])
                        distance = target_hash_obj - stored_hash_obj
                        similarities[hash_type] = distance
                    except Exception:
                        similarities[hash_type] = float('inf')
            
            # Calculate average similarity
            if similarities:
                avg_distance = np.mean(list(similarities.values()))
                if avg_distance <= self.similarity_threshold:
                    similar_hashes.append({
                        'filename': stored_filename,
                        'average_distance': avg_distance,
                        'hash_distances': similarities,
                        'hashes': stored_hashes
                    })
        
        # Sort by similarity (lower distance = more similar)
        similar_hashes.sort(key=lambda x: x['average_distance'])
        return similar_hashes
    
    def add_to_database(self, filename: str, hashes: Dict[str, str]):
        """Add new hashes to the database"""
        self.hash_database["hashes"][filename] = hashes
        self.hash_database["metadata"]["total_files"] += 1
        self._save_hash_database()
    
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """Analyze image for duplicates"""
        try:
            # Generate hashes for the uploaded image
            hashes = self.generate_hashes(image_path)
            
            # Find similar images
            similar_images = self.find_similar_hashes(hashes)
            
            # Determine if it's a duplicate
            is_duplicate = len(similar_images) > 0
            confidence = 0.0
            
            if is_duplicate:
                # Use the best match for confidence calculation
                best_match = similar_images[0]
                max_distance = max(self.similarity_threshold, best_match['average_distance'])
                confidence = 1.0 - (best_match['average_distance'] / max_distance)
                confidence = max(0.0, min(1.0, confidence))
            else:
                confidence = 0.95  # High confidence that it's not a duplicate
            
            # Add to database if it's not a duplicate
            filename = os.path.basename(image_path)
            if not is_duplicate:
                self.add_to_database(filename, hashes)
            
            return {
                'hash_value': hashes.get('perceptual_hash', ''),
                'all_hashes': hashes,
                'is_duplicate': is_duplicate,
                'confidence': confidence,
                'similar_hashes': [img['hashes']['perceptual_hash'] for img in similar_images[:5]],
                'similar_files': [img['filename'] for img in similar_images[:5]],
                'similarity_details': similar_images[:3] if similar_images else []
            }
            
        except Exception as e:
            return {
                'hash_value': '',
                'all_hashes': {},
                'is_duplicate': False,
                'confidence': 0.0,
                'similar_hashes': [],
                'similar_files': [],
                'similarity_details': [],
                'error': str(e)
            }
    
    def get_database_stats(self) -> Dict:
        """Get statistics about the hash database"""
        return {
            'total_files': self.hash_database["metadata"]["total_files"],
            'database_size': len(self.hash_database["hashes"]),
            'hash_types': ['average_hash', 'perceptual_hash', 'difference_hash', 'wavelet_hash']
        }
