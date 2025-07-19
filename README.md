# Hand Sign Recognition ML System

A real-time machine learning system for recognizing hand signs using computer vision and deep learning.

## Features

- **Real-time Data Collection**: Capture hand sign data directly from your camera
- **Advanced ML Model**: Deep learning model using CNN architecture for accurate classification
- **Modern GUI**: User-friendly interface for data collection, training, and prediction
- **Live Recognition**: Real-time hand sign recognition with visual feedback
- **Model Training**: Train custom models with your own hand sign datasets

## Requirements

- Python 3.8+
- OpenCV
- MediaPipe
- TensorFlow
- NumPy
- Tkinter (for GUI)
- Pillow (for image processing)

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python main.py
```

## Usage

1. **Data Collection**: Use the GUI to collect hand sign data for different gestures
2. **Model Training**: Train the ML model with your collected data
3. **Real-time Recognition**: Use the trained model for live hand sign recognition

## Project Structure

```
hand-sign-recognition/
├── src/
│   ├── data_collection.py    # Camera data collection module
│   ├── model.py             # ML model definition and training
│   ├── preprocessor.py      # Data preprocessing utilities
│   ├── gui.py              # Modern GUI interface
│   └── utils.py            # Helper functions
├── models/                 # Saved ML models
├── data/                   # Training data storage
├── main.py                 # Main application entry point
├── requirements.txt        # Project dependencies
└── README.md              # This file
```

## Model Architecture

The system uses a Convolutional Neural Network (CNN) with:
- Hand landmark detection using MediaPipe
- Feature extraction from hand keypoints
- Deep learning classification
- Real-time inference optimization

## Contributing

Feel free to contribute by improving the model accuracy, adding new features, or enhancing the GUI.
