import numpy as np
import cv2
import os
import json
from typing import List, Dict, Tuple, Optional
import time
from datetime import datetime

class HandSignUtils:
    """
    Utility functions for hand sign recognition system.
    """
    
    @staticmethod
    def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize hand landmarks for consistent processing.
        
        Args:
            landmarks: Raw landmark coordinates (63 values: 21 landmarks * 3 coords)
            
        Returns:
            Normalized landmarks
        """
        # Reshape to (21, 3) for easier processing
        landmarks_reshaped = landmarks.reshape(21, 3)
        
        # Calculate wrist position (landmark 0)
        wrist = landmarks_reshaped[0]
        
        # Translate all landmarks relative to wrist
        normalized = landmarks_reshaped - wrist
        
        # Calculate scale based on hand size (distance from wrist to middle finger tip)
        middle_finger_tip = normalized[12]  # Middle finger tip
        scale = np.linalg.norm(middle_finger_tip)
        
        # Normalize by scale if scale is not zero
        if scale > 0:
            normalized = normalized / scale
        
        # Flatten back to 1D array
        return normalized.flatten()
    
    @staticmethod
    def augment_landmarks(landmarks: np.ndarray, num_augmentations: int = 5) -> List[np.ndarray]:
        """
        Generate augmented versions of hand landmarks for data augmentation.
        
        Args:
            landmarks: Original landmark coordinates
            num_augmentations: Number of augmented versions to generate
            
        Returns:
            List of augmented landmark arrays
        """
        augmented = []
        landmarks_reshaped = landmarks.reshape(21, 3)
        
        for _ in range(num_augmentations):
            # Add small random noise
            noise = np.random.normal(0, 0.01, landmarks_reshaped.shape)
            noisy_landmarks = landmarks_reshaped + noise
            
            # Random scaling (±10%)
            scale_factor = np.random.uniform(0.9, 1.1)
            scaled_landmarks = noisy_landmarks * scale_factor
            
            # Random rotation around Z-axis
            angle = np.random.uniform(-0.1, 0.1)  # Small rotation
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ])
            
            rotated_landmarks = np.dot(scaled_landmarks, rotation_matrix.T)
            augmented.append(rotated_landmarks.flatten())
        
        return augmented
    
    @staticmethod
    def calculate_hand_features(landmarks: np.ndarray) -> Dict[str, float]:
        """
        Calculate additional hand features for improved recognition.
        
        Args:
            landmarks: Hand landmark coordinates
            
        Returns:
            Dictionary of calculated features
        """
        landmarks_reshaped = landmarks.reshape(21, 3)
        
        features = {}
        
        # Finger tip positions (relative to wrist)
        wrist = landmarks_reshaped[0]
        finger_tips = {
            'thumb': landmarks_reshaped[4] - wrist,
            'index': landmarks_reshaped[8] - wrist,
            'middle': landmarks_reshaped[12] - wrist,
            'ring': landmarks_reshaped[16] - wrist,
            'pinky': landmarks_reshaped[20] - wrist
        }
        
        # Finger lengths
        for finger, tip in finger_tips.items():
            features[f'{finger}_length'] = np.linalg.norm(tip)
        
        # Angles between fingers
        thumb_vec = finger_tips['thumb']
        index_vec = finger_tips['index']
        middle_vec = finger_tips['middle']
        
        # Angle between thumb and index finger
        if np.linalg.norm(thumb_vec) > 0 and np.linalg.norm(index_vec) > 0:
            cos_angle = np.dot(thumb_vec, index_vec) / (np.linalg.norm(thumb_vec) * np.linalg.norm(index_vec))
            features['thumb_index_angle'] = np.arccos(np.clip(cos_angle, -1, 1))
        
        # Palm area approximation
        palm_points = landmarks_reshaped[[0, 5, 9, 13, 17]]  # Wrist and base of fingers
        features['palm_area'] = HandSignUtils._calculate_polygon_area(palm_points[:, :2])
        
        # Hand orientation
        if np.linalg.norm(middle_vec) > 0:
            features['hand_orientation'] = np.arctan2(middle_vec[1], middle_vec[0])
        
        return features
    
    @staticmethod
    def _calculate_polygon_area(points: np.ndarray) -> float:
        """Calculate area of a polygon using shoelace formula."""
        n = len(points)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]
        return abs(area) / 2.0
    
    @staticmethod
    def merge_datasets(dataset_paths: List[str], output_path: str) -> bool:
        """
        Merge multiple hand sign datasets into one.
        
        Args:
            dataset_paths: List of paths to dataset JSON files
            output_path: Path to save merged dataset
            
        Returns:
            True if successful, False otherwise
        """
        try:
            merged_data = []
            total_samples = 0
            
            for path in dataset_paths:
                with open(path, 'r') as f:
                    data = json.load(f)
                
                samples = data.get('data', [])
                merged_data.extend(samples)
                total_samples += len(samples)
                print(f"Loaded {len(samples)} samples from {path}")
            
            # Create merged dataset
            merged_dataset = {
                'metadata': {
                    'total_samples': total_samples,
                    'merge_date': datetime.now().isoformat(),
                    'source_files': dataset_paths,
                    'landmark_count': 63
                },
                'data': merged_data
            }
            
            # Save merged dataset
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(merged_dataset, f, indent=2)
            
            print(f"Merged dataset saved to {output_path} with {total_samples} samples")
            return True
            
        except Exception as e:
            print(f"Error merging datasets: {e}")
            return False
    
    @staticmethod
    def validate_dataset(dataset_path: str) -> Dict[str, any]:
        """
        Validate a hand sign dataset and return statistics.
        
        Args:
            dataset_path: Path to dataset JSON file
            
        Returns:
            Dictionary with validation results
        """
        try:
            with open(dataset_path, 'r') as f:
                data = json.load(f)
            
            samples = data.get('data', [])
            
            # Basic statistics
            total_samples = len(samples)
            labels = [sample['label'] for sample in samples]
            unique_labels = list(set(labels))
            
            # Label distribution
            label_counts = {}
            for label in labels:
                label_counts[label] = label_counts.get(label, 0) + 1
            
            # Check for data quality issues
            issues = []
            
            # Check for missing landmarks
            invalid_samples = 0
            for sample in samples:
                landmarks = sample.get('landmarks', [])
                if len(landmarks) != 63:
                    invalid_samples += 1
            
            if invalid_samples > 0:
                issues.append(f"{invalid_samples} samples have invalid landmark count")
            
            # Check for balanced dataset
            min_samples = min(label_counts.values()) if label_counts else 0
            max_samples = max(label_counts.values()) if label_counts else 0
            imbalance_ratio = max_samples / min_samples if min_samples > 0 else float('inf')
            
            if imbalance_ratio > 3:
                issues.append(f"Dataset is imbalanced (ratio: {imbalance_ratio:.2f})")
            
            # Minimum samples per class
            if min_samples < 50:
                issues.append(f"Some classes have too few samples (min: {min_samples})")
            
            return {
                'valid': len(issues) == 0,
                'total_samples': total_samples,
                'unique_labels': len(unique_labels),
                'label_distribution': label_counts,
                'issues': issues,
                'imbalance_ratio': imbalance_ratio,
                'invalid_samples': invalid_samples
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': str(e)
            }
    
    @staticmethod
    def create_sample_config() -> Dict[str, any]:
        """
        Create a sample configuration for the hand sign recognition system.
        
        Returns:
            Dictionary with default configuration
        """
        return {
            'camera': {
                'width': 640,
                'height': 480,
                'fps': 30,
                'device_index': 0
            },
            'data_collection': {
                'default_duration': 5,
                'min_samples_per_class': 50,
                'auto_save': True,
                'save_interval': 100
            },
            'model': {
                'architecture': 'dense',
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 50,
                'validation_split': 0.2,
                'early_stopping_patience': 15
            },
            'recognition': {
                'confidence_threshold': 0.7,
                'smoothing_window': 5,
                'top_k_predictions': 3
            },
            'gui': {
                'theme': 'dark',
                'update_interval': 33,  # ~30 FPS
                'window_size': '1200x800'
            }
        }
    
    @staticmethod
    def save_config(config: Dict[str, any], path: str = "config.json") -> bool:
        """
        Save configuration to file.
        
        Args:
            config: Configuration dictionary
            path: Path to save configuration
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(path, 'w') as f:
                json.dump(config, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
    
    @staticmethod
    def load_config(path: str = "config.json") -> Optional[Dict[str, any]]:
        """
        Load configuration from file.
        
        Args:
            path: Path to configuration file
            
        Returns:
            Configuration dictionary or None if failed
        """
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return None
    
    @staticmethod
    def benchmark_model(model, test_data_path: str) -> Dict[str, float]:
        """
        Benchmark model performance.
        
        Args:
            model: Trained model instance
            test_data_path: Path to test data
            
        Returns:
            Dictionary with performance metrics
        """
        try:
            # Load test data
            with open(test_data_path, 'r') as f:
                data = json.load(f)
            
            samples = data['data']
            
            # Prepare test data
            X_test = []
            y_true = []
            
            for sample in samples:
                landmarks = np.array(sample['landmarks'])
                label = sample['label']
                
                X_test.append(landmarks)
                y_true.append(label)
            
            X_test = np.array(X_test)
            
            # Measure inference time
            start_time = time.time()
            predictions = []
            
            for landmarks in X_test:
                pred = model.predict(landmarks, top_k=1)
                if pred:
                    predictions.append(pred[0][0])
                else:
                    predictions.append("unknown")
            
            end_time = time.time()
            inference_time = (end_time - start_time) / len(X_test)
            
            # Calculate accuracy
            correct = sum(1 for true, pred in zip(y_true, predictions) if true == pred)
            accuracy = correct / len(y_true) if y_true else 0
            
            return {
                'accuracy': accuracy,
                'inference_time_ms': inference_time * 1000,
                'fps': 1 / inference_time if inference_time > 0 else 0,
                'total_samples': len(X_test)
            }
            
        except Exception as e:
            print(f"Benchmarking error: {e}")
            return {'error': str(e)}

# Example usage and testing functions
if __name__ == "__main__":
    # Create sample config
    config = HandSignUtils.create_sample_config()
    HandSignUtils.save_config(config)
    
    print("Sample configuration created and saved to config.json")
    print("Available utility functions:")
    print("- normalize_landmarks(): Normalize hand landmarks")
    print("- augment_landmarks(): Generate augmented training data")
    print("- calculate_hand_features(): Extract additional features")
    print("- merge_datasets(): Combine multiple datasets")
    print("- validate_dataset(): Check dataset quality")
    print("- benchmark_model(): Test model performance")
