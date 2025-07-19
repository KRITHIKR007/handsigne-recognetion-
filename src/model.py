import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import pickle

class HandSignModel:
    """
    Deep learning model for hand sign recognition using TensorFlow/Keras.
    Implements a CNN architecture optimized for hand landmark classification.
    """
    
    def __init__(self, num_classes: int = None):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.num_classes = num_classes
        self.input_shape = (63,)  # 21 landmarks * 3 coordinates (x, y, z)
        self.history = None
        self.is_trained = False
        
    def create_model(self, num_classes: int) -> keras.Model:
        """
        Create the neural network architecture for hand sign classification.
        
        Args:
            num_classes: Number of different hand signs to classify
            
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # Dense layers with dropout for regularization
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        return model
    
    def prepare_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and prepare data for training.
        
        Args:
            data_path: Path to the JSON data file
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Load data from JSON file
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        samples = data['data']
        
        # Extract features and labels
        X = []
        y = []
        
        for sample in samples:
            landmarks = np.array(sample['landmarks'])
            label = sample['label']
            
            # Normalize landmarks (optional, already normalized by MediaPipe)
            landmarks = self._normalize_landmarks(landmarks)
            
            X.append(landmarks)
            y.append(label)
        
        X = np.array(X)
        y = np.array(y)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        y_categorical = keras.utils.to_categorical(y_encoded)
        
        # Update number of classes
        self.num_classes = len(self.label_encoder.classes_)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print(f"Data prepared: {len(X_train)} training samples, {len(X_test)} test samples")
        print(f"Classes: {list(self.label_encoder.classes_)}")
        
        return X_train, X_test, y_train, y_test
    
    def _normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize hand landmarks for better model performance.
        
        Args:
            landmarks: Raw landmark coordinates
            
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
    
    def train(self, data_path: str, epochs: int = 100, batch_size: int = 32, 
              validation_split: float = 0.2) -> dict:
        """
        Train the hand sign recognition model.
        
        Args:
            data_path: Path to training data JSON file
            epochs: Number of training epochs
            batch_size: Training batch size
            validation_split: Fraction of data to use for validation
            
        Returns:
            Training history dictionary
        """
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(data_path)
        
        # Create model
        self.model = self.create_model(self.num_classes)
        
        print("\nModel Architecture:")
        self.model.summary()
        
        # Callbacks for training
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                'models/best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train the model
        print("\nStarting training...")
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_loss, test_accuracy, test_top_k = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Top-K Accuracy: {test_top_k:.4f}")
        
        # Generate classification report
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        print("\nClassification Report:")
        print(classification_report(
            y_true_classes, y_pred_classes,
            target_names=self.label_encoder.classes_
        ))
        
        self.is_trained = True
        
        return {
            'history': self.history.history,
            'test_accuracy': test_accuracy,
            'test_top_k_accuracy': test_top_k
        }
    
    def predict(self, landmarks: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Predict hand sign from landmarks.
        
        Args:
            landmarks: Hand landmark coordinates
            top_k: Number of top predictions to return
            
        Returns:
            List of (label, confidence) tuples
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Normalize landmarks
        normalized_landmarks = self._normalize_landmarks(landmarks)
        
        # Reshape for prediction
        input_data = normalized_landmarks.reshape(1, -1)
        
        # Make prediction
        predictions = self.model.predict(input_data, verbose=0)[0]
        
        # Get top-k predictions
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            label = self.label_encoder.inverse_transform([idx])[0]
            confidence = predictions[idx]
            results.append((label, confidence))
        
        return results
    
    def save_model(self, model_path: str = "models/hand_sign_model.h5", 
                   encoder_path: str = "models/label_encoder.pkl"):
        """
        Save the trained model and label encoder.
        
        Args:
            model_path: Path to save the model
            encoder_path: Path to save the label encoder
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before saving")
        
        # Create models directory
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model
        self.model.save(model_path)
        
        # Save label encoder
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        print(f"Model saved to {model_path}")
        print(f"Label encoder saved to {encoder_path}")
    
    def load_model(self, model_path: str = "models/hand_sign_model.h5", 
                   encoder_path: str = "models/label_encoder.pkl"):
        """
        Load a pre-trained model and label encoder.
        
        Args:
            model_path: Path to the saved model
            encoder_path: Path to the saved label encoder
        """
        try:
            # Load model
            self.model = keras.models.load_model(model_path)
            
            # Load label encoder
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            self.num_classes = len(self.label_encoder.classes_)
            self.is_trained = True
            
            print(f"Model loaded from {model_path}")
            print(f"Label encoder loaded from {encoder_path}")
            print(f"Classes: {list(self.label_encoder.classes_)}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def plot_training_history(self, save_path: str = "models/training_history.png"):
        """
        Plot training history graphs.
        
        Args:
            save_path: Path to save the plot
        """
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        # Top-K Accuracy
        if 'top_k_categorical_accuracy' in self.history.history:
            axes[1, 0].plot(self.history.history['top_k_categorical_accuracy'], label='Training Top-K')
            axes[1, 0].plot(self.history.history['val_top_k_categorical_accuracy'], label='Validation Top-K')
            axes[1, 0].set_title('Top-K Accuracy')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Top-K Accuracy')
            axes[1, 0].legend()
        
        # Learning Rate (if available)
        if hasattr(self.model.optimizer, 'learning_rate'):
            axes[1, 1].plot(self.history.history.get('lr', []))
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Training history plot saved to {save_path}")

if __name__ == "__main__":
    # Example usage
    model = HandSignModel()
    
    # Train the model (assuming data file exists)
    data_file = "data/hand_sign_data.json"
    if os.path.exists(data_file):
        results = model.train(data_file, epochs=50)
        model.save_model()
        model.plot_training_history()
    else:
        print(f"Data file {data_file} not found. Please collect data first.")
