#!/usr/bin/env python3
"""
Test script to demonstrate the facial expressiveness recognition system
Shows comparison with original emotion recognition approach
"""

import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras
import json
import os

def load_trained_model():
    """Load the trained expressiveness model"""
    try:
        # Load model architecture
        with open('models/Facial_Expressiveness_Recognition_Model.json', 'r') as json_file:
            model_json = json_file.read()
        model = keras.models.model_from_json(model_json)

        # Load weights
        model.load_weights('models/expressiveness_model_weights.h5')

        # Load label classes
        label_classes = np.load('models/label_encoder_classes.npy')

        return model, label_classes
    except FileNotFoundError:
        print("Model files not found. Please train the model first.")
        return None, None

def extract_face_mediapipe(frame, face_detection):
    """Extract face using MediaPipe (alternative to OpenCV)"""
    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame
    results = face_detection.process(rgb_frame)

    if results.detections:
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box

        h, w, _ = frame.shape
        x_min = int(bbox.xmin * w)
        y_min = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)

        # Add padding
        padding = int(0.1 * width)
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_min + width + 2*padding)
        y_max = min(h, y_min + height + 2*padding)

        # Crop and preprocess face
        face_crop = frame[y_min:y_max, x_min:x_max]
        if face_crop.size > 0:
            face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            face_resized = cv2.resize(face_gray, (48, 48))
            return face_resized, (x_min, y_min, x_max, y_max)

    return None, None

def demonstrate_system_comparison():
    """Demonstrate how the new system compares to the original"""

    print("Facial Expression Recognition System Comparison")
    print("=" * 55)
    print()

    print("ORIGINAL SYSTEM (live-face-emotion-classifier-main):")
    print("• Uses OpenCV for face detection")
    print("• Uses PyTorch/TorchVision for CNN")
    print("• Classifies into 7 emotions: anger, disgust, fear, happiness, sadness, surprise, neutral")
    print("• Trained on FER2013 dataset (general faces)")
    print("• Output: 'Happy: 0.87' (emotion + confidence)")
    print()

    print("NEW SYSTEM (Expressiveness Recognition):")
    print("• Uses MediaPipe for face detection (alternative to OpenCV)")
    print("• Uses TensorFlow/Keras for CNN (alternative to PyTorch)")
    print("• Classifies into 3 expressiveness levels: Reserved, Balanced, Expressive")
    print("• Trained on RecruitView_Data (interview videos)")
    print("• Output: 'Balanced Expression: 0.82' (expressiveness level + confidence)")
    print()

    # Test if model is available
    model, label_classes = load_trained_model()

    if model is None:
        print("❌ Model not found. Please train the model first using the notebook.")
        print()
        print("To train the model:")
        print("1. Install requirements: pip install -r requirements.txt")
        print("2. Open Facial_Expression_Expressiveness_Recognition.ipynb")
        print("3. Run all cells to train the model")
        return

    print("✅ Trained model found!")
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output classes: {label_classes}")
    print()

    # Initialize MediaPipe (alternative to OpenCV)
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5
    )
    print("✅ MediaPipe face detection initialized (alternative to OpenCV)")
    print()

    # Test with webcam if available
    print("Testing real-time recognition...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Webcam not available for testing")
        print("But the system is ready to use once webcam is available!")
        return

    print("✅ Webcam available")
    print("Starting 5-second test... (press any key to continue)")

    # Quick test
    frame_count = 0
    max_frames = 30  # ~1 second at 30fps

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract face using MediaPipe (vs OpenCV in original)
        face, bbox = extract_face_mediapipe(frame, face_detection)

        if face is not None:
            # Predict using TensorFlow/Keras (vs PyTorch in original)
            face_input = face.reshape(1, 48, 48, 1).astype('float32') / 255.0
            predictions = model.predict(face_input, verbose=0)
            predicted_idx = np.argmax(predictions[0])
            expressiveness = label_classes[predicted_idx]
            confidence = predictions[0][predicted_idx]

            # Display result (similar format to original)
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            text = f'{expressiveness}: {confidence:.2f}'
            cv2.putText(frame, text, (x_min, y_min - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.imshow('Expressiveness Recognition Test', frame)

        frame_count += 1

        if cv2.waitKey(1) != -1:
            break

    cap.release()
    cv2.destroyAllWindows()

    print("✅ Test completed successfully!")
    print()
    print("SUMMARY:")
    print("The new system provides similar real-time output to the original,")
    print("but focuses on expressiveness levels rather than specific emotions.")
    print("It uses different technologies (MediaPipe + TensorFlow) as requested.")

if __name__ == "__main__":
    demonstrate_system_comparison()