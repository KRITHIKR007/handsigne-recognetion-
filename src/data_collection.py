import cv2
import mediapipe as mp
import numpy as np
import os
import json
from datetime import datetime
import threading
from typing import List, Tuple, Optional

class HandSignDataCollector:
    """
    Real-time hand sign data collection using MediaPipe and OpenCV.
    Captures hand landmarks and saves them for model training.
    """
    
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Data storage
        self.collected_data = []
        self.current_label = ""
        self.is_collecting = False
        self.frame_count = 0
        
        # Camera setup
        self.cap = None
        self.camera_active = False
        
    def start_camera(self, camera_index: int = 0) -> bool:
        """Start the camera capture."""
        try:
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                return False
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.camera_active = True
            return True
        except Exception as e:
            print(f"Error starting camera: {e}")
            return False
    
    def stop_camera(self):
        """Stop the camera capture."""
        if self.cap:
            self.cap.release()
        self.camera_active = False
    
    def extract_hand_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract hand landmarks from a frame using MediaPipe.
        
        Args:
            frame: Input image frame
            
        Returns:
            Normalized hand landmarks array or None if no hand detected
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            # Get the first hand landmarks
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Extract landmark coordinates
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            return np.array(landmarks)
        
        return None
    
    def draw_landmarks(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw hand landmarks on the frame for visualization.
        
        Args:
            frame: Input image frame
            
        Returns:
            Frame with drawn landmarks
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        return frame
    
    def start_collection(self, label: str, duration: int = 5):
        """
        Start collecting data for a specific hand sign label.
        
        Args:
            label: The label for the hand sign being collected
            duration: Duration in seconds to collect data
        """
        self.current_label = label
        self.is_collecting = True
        self.frame_count = 0
        
        print(f"Starting data collection for '{label}' for {duration} seconds...")
        
        # Start collection timer
        timer = threading.Timer(duration, self.stop_collection)
        timer.start()
    
    def stop_collection(self):
        """Stop data collection."""
        self.is_collecting = False
        print(f"Data collection stopped. Collected {len(self.collected_data)} samples for '{self.current_label}'")
    
    def get_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get the current frame from camera and process it.
        
        Returns:
            Tuple of (processed_frame, landmarks) or (None, None) if no frame
        """
        if not self.camera_active or not self.cap:
            return None, None
        
        ret, frame = self.cap.read()
        if not ret:
            return None, None
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Extract landmarks
        landmarks = self.extract_hand_landmarks(frame)
        
        # If collecting data and landmarks detected
        if self.is_collecting and landmarks is not None:
            self.collected_data.append({
                'landmarks': landmarks.tolist(),
                'label': self.current_label,
                'timestamp': datetime.now().isoformat(),
                'frame_id': self.frame_count
            })
            self.frame_count += 1
        
        # Draw landmarks on frame
        frame_with_landmarks = self.draw_landmarks(frame.copy())
        
        # Add status text and visual feedback
        status = "COLLECTING" if self.is_collecting else "READY"
        color = (0, 0, 255) if self.is_collecting else (0, 255, 0)
        
        # Main status
        cv2.putText(frame_with_landmarks, f"Status: {status}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        if self.is_collecting:
            # Collection info
            cv2.putText(frame_with_landmarks, f"Label: {self.current_label}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame_with_landmarks, f"Samples: {len(self.collected_data)}", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # Visual collection indicator (pulsing border)
            border_thickness = 10
            h, w = frame_with_landmarks.shape[:2]
            cv2.rectangle(frame_with_landmarks, (0, 0), (w, h), (0, 0, 255), border_thickness)
        
        # Hand detection feedback
        if landmarks is not None:
            cv2.putText(frame_with_landmarks, "HAND DETECTED", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Show hand confidence area
            cv2.circle(frame_with_landmarks, (50, 180), 20, (0, 255, 0), -1)
        else:
            cv2.putText(frame_with_landmarks, "Show your hand", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Show instruction area
            cv2.circle(frame_with_landmarks, (50, 180), 20, (0, 255, 255), 2)
        
        return frame_with_landmarks, landmarks
    
    def save_data(self, filename: str = None) -> str:
        """
        Save collected data to a JSON file.
        
        Args:
            filename: Optional filename, if None generates timestamp-based name
            
        Returns:
            Path to saved file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hand_sign_data_{timestamp}.json"
        
        filepath = os.path.join("data", filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        data_to_save = {
            'metadata': {
                'total_samples': len(self.collected_data),
                'collection_date': datetime.now().isoformat(),
                'landmark_count': 63  # 21 landmarks * 3 coordinates
            },
            'data': self.collected_data
        }
        
        with open(filepath, 'w') as f:
            json.dump(data_to_save, f, indent=2)
        
        print(f"Data saved to {filepath}")
        return filepath
    
    def load_data(self, filepath: str) -> bool:
        """
        Load previously collected data from a JSON file.
        
        Args:
            filepath: Path to the data file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.collected_data = data.get('data', [])
            print(f"Loaded {len(self.collected_data)} samples from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def clear_data(self):
        """Clear all collected data."""
        self.collected_data = []
        # Also clear the new format data if it exists
        if hasattr(self, 'landmarks_data'):
            self.landmarks_data = []
        if hasattr(self, 'labels_data'):
            self.labels_data = []
        print("Collected data cleared")
    
    def has_data(self) -> bool:
        """Check if any data has been collected."""
        if hasattr(self, 'landmarks_data') and self.landmarks_data:
            return len(self.landmarks_data) > 0
        return len(self.collected_data) > 0
    
    def add_sample(self, label: str, landmarks: np.ndarray):
        """Add a single sample to the dataset."""
        if not hasattr(self, 'landmarks_data'):
            self.landmarks_data = []
        if not hasattr(self, 'labels_data'):
            self.labels_data = []
        
        self.landmarks_data.append(landmarks)
        self.labels_data.append(label)
    
    def get_current_landmarks(self) -> Optional[np.ndarray]:
        """Get landmarks from the current frame."""
        frame_data = self.get_frame()
        if frame_data is None:
            return None
        
        # Handle tuple return from get_frame()
        if isinstance(frame_data, tuple):
            frame, landmarks = frame_data
            # Return the landmarks if they exist
            if landmarks is not None:
                return landmarks
            # If no landmarks in the tuple, extract them from the frame
            if frame is None:
                return None
        else:
            frame = frame_data
        
        # Process frame to extract landmarks
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
            return np.array(landmarks)
        
        return None
    
    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Get training data in format suitable for ML model."""
        if not hasattr(self, 'landmarks_data') or not self.landmarks_data:
            # Convert old format if needed
            if self.collected_data:
                landmarks_list = []
                labels_list = []
                for item in self.collected_data:
                    landmarks_list.append(item['landmarks'])
                    labels_list.append(item['label'])
                
                # Get unique labels
                unique_labels = list(set(labels_list))
                
                # Convert to numpy arrays
                X = np.array(landmarks_list)
                y = np.array([unique_labels.index(label) for label in labels_list])
                
                return X, y, unique_labels
            else:
                return np.array([]), np.array([]), []
        
        # Use new format
        unique_labels = list(set(self.labels_data))
        X = np.array(self.landmarks_data)
        y = np.array([unique_labels.index(label) for label in self.labels_data])
        
        return X, y, unique_labels
    
    def get_data_summary(self) -> dict:
        """
        Get summary of collected data.
        
        Returns:
            Dictionary with data statistics
        """
        if not self.collected_data:
            return {'total_samples': 0, 'labels': {}}
        
        label_counts = {}
        for item in self.collected_data:
            label = item['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        return {
            'total_samples': len(self.collected_data),
            'labels': label_counts,
            'unique_labels': len(label_counts)
        }
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.stop_camera()
        if hasattr(self, 'hands'):
            self.hands.close()

if __name__ == "__main__":
    # Example usage
    collector = HandSignDataCollector()
    
    if collector.start_camera():
        print("Camera started. Press 'q' to quit, 'c' to collect data, 's' to save")
        
        while True:
            frame, landmarks = collector.get_frame()
            
            if frame is not None:
                cv2.imshow('Hand Sign Data Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                label = input("Enter label for hand sign: ")
                collector.start_collection(label, 5)
            elif key == ord('s'):
                collector.save_data()
        
        collector.stop_camera()
        cv2.destroyAllWindows()
    else:
        print("Failed to start camera")
