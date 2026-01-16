#!/usr/bin/env python3
"""
Real-time Facial Expressiveness Recognition Demo
Similar to the original live-face-emotion-classifier but with expressiveness categories
"""

import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras
import json
import os

class FacialExpressivenessRecognizer:
    def __init__(self, model_path=None, weights_path=None, labels_path=None):
        """
        Initialize the facial expressiveness recognizer

        Args:
            model_path: Path to model JSON file
            weights_path: Path to model weights file
            labels_path: Path to label classes file
        """
        # Default paths
        if model_path is None:
            model_path = 'models/Facial_Expressiveness_Recognition_Model.json'
        if weights_path is None:
            weights_path = 'models/expressiveness_model_weights.h5'
        if labels_path is None:
            labels_path = 'models/label_encoder_classes.npy'

        # Load model
        try:
            with open(model_path, 'r') as json_file:
                model_json = json_file.read()
            self.model = keras.models.model_from_json(model_json)
            self.model.load_weights(weights_path)
            print("Model loaded successfully!")
        except FileNotFoundError:
            print("Model files not found. Please train the model first using the notebook.")
            self.model = None
            return

        # Load label classes
        try:
            self.label_classes = np.load(labels_path)
            print(f"Label classes: {self.label_classes}")
        except FileNotFoundError:
            print("Label classes file not found.")
            self.label_classes = ['Reserved Expression', 'Balanced Expression', 'Expressive']

        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )

    def extract_face(self, frame):
        """Extract face from frame using MediaPipe"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame
        results = self.face_detection.process(rgb_frame)

        if results.detections:
            # Get the first detected face
            detection = results.detections[0]

            # Get bounding box
            bbox = detection.location_data.relative_bounding_box

            # Convert relative coordinates to absolute
            h, w, _ = frame.shape
            x_min = int(bbox.xmin * w)
            y_min = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            # Add some padding
            padding = int(0.1 * width)
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_min + width + 2*padding)
            y_max = min(h, y_min + height + 2*padding)

            # Crop face
            face_crop = frame[y_min:y_max, x_min:x_max]

            if face_crop.size > 0:
                # Convert to grayscale
                face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

                # Resize to 48x48
                face_resized = cv2.resize(face_gray, (48, 48))

                return face_resized, (x_min, y_min, x_max, y_max)

        return None, None

    def predict_expressiveness(self, face_image):
        """Predict facial expressiveness from face image"""
        if self.model is None:
            return "Model not loaded", 0.0

        # Preprocess image
        face_input = face_image.reshape(1, 48, 48, 1).astype('float32') / 255.0

        # Predict
        predictions = self.model.predict(face_input, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = self.label_classes[predicted_class_idx]
        confidence = predictions[0][predicted_class_idx]

        return predicted_class, confidence

    def run_real_time_recognition(self):
        """Run real-time facial expressiveness recognition"""
        if self.model is None:
            print("Model not loaded. Cannot run real-time recognition.")
            return

        # Initialize webcam
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Cannot open webcam")
            return

        print("Starting real-time facial expressiveness recognition...")
        print("Press 'q' to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Extract face
            face, bbox = self.extract_face(frame)

            if face is not None and bbox is not None:
                # Predict expressiveness
                expressiveness, confidence = self.predict_expressiveness(face)

                # Draw bounding box
                x_min, y_min, x_max, y_max = bbox
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Display result
                text = f'{expressiveness}: {confidence:.2f}'
                cv2.putText(frame, text, (x_min, y_min - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Display frame
            cv2.imshow('Facial Expressiveness Recognition', frame)

            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def main():
    """Main function to demonstrate the facial expressiveness recognizer"""
    print("Facial Expressiveness Recognition System")
    print("=" * 45)
    print("This system classifies facial expressions into 3 categories:")
    print("- Reserved Expression: Low facial expressiveness")
    print("- Balanced Expression: Neutral facial expressiveness")
    print("- Expressive: High facial expressiveness")
    print()

    # Initialize recognizer
    recognizer = FacialExpressivenessRecognizer()

    if recognizer.model is None:
        print("Model not found. Please train the model first using the notebook.")
        print("Steps to train the model:")
        print("1. Install requirements: pip install -r requirements.txt")
        print("2. Open Facial_Expression_Expressiveness_Recognition.ipynb")
        print("3. Run all cells to train the model")
        return

    # Run real-time recognition
    try:
        recognizer.run_real_time_recognition()
    except KeyboardInterrupt:
        print("Recognition stopped by user")
    except Exception as e:
        print(f"Error during recognition: {e}")

if __name__ == "__main__":
    main()