# Facial Expression Expressiveness Recognition System - Usage Guide

## Overview

This system is a new facial expression recognition approach that **differs from traditional emotion classification**. Instead of classifying into 7 basic emotions (like Ekman's system), it focuses on **facial expressiveness levels** during interviews.

## Key Differences from Original System

| Aspect | Original System (live-face-emotion-classifier-main) | New System (Expressiveness Recognition) |
|--------|---------------------------------------------------|-----------------------------------------|
| **Emotions** | 7 emotions: anger, disgust, fear, happiness, sadness, surprise, neutral | 3 expressiveness levels: Reserved, Balanced, Expressive |
| **Face Detection** | OpenCV Haar cascades | MediaPipe (Google's framework) |
| **Deep Learning** | PyTorch + TorchVision | TensorFlow/Keras |
| **Dataset** | FER2013 (general faces) | RecruitView_Data (interview videos) |
| **Purpose** | General emotion recognition | Interview facial expressiveness analysis |

## Installation

1. **Install Python packages:**
```bash
pip install -r requirements.txt
```

2. **Required packages:**
- tensorflow (instead of PyTorch)
- mediapipe (instead of OpenCV)
- pandas, numpy, matplotlib, seaborn, scikit-learn, tqdm

## Usage Options

### Option 1: Complete Training and Recognition (Recommended)

1. **Open the main notebook:**
```bash
# Open Facial_Expression_Expressiveness_Recognition.ipynb in Jupyter
jupyter notebook Facial_Expression_Expressiveness_Recognition.ipynb
```

2. **Run all cells in order:**
   - Data loading and analysis
   - Face extraction from videos
   - Model training
   - Evaluation and saving

### Option 2: Quick Real-time Demo (After Training)

If you already have a trained model, use the demo script:

```bash
python real_time_demo.py
```

## System Components

### 1. Expressiveness Categories

Instead of emotions, the system classifies facial expressions into:

- **Reserved Expression** (Low expressiveness): Facial expression score ≤ -0.303
- **Balanced Expression** (Neutral expressiveness): -0.303 < score ≤ 0.294
- **Expressive** (High expressiveness): score > 0.294

### 2. Data Processing

- **Input**: Video files from RecruitView_Data
- **Processing**: Extract faces from video frames using MediaPipe
- **Output**: 48x48 grayscale face images with expressiveness labels

### 3. Model Architecture

- **Input**: 48x48 grayscale images
- **Layers**: 3 Conv2D + BatchNorm + Dropout blocks
- **Output**: 3 classes (softmax)
- **Framework**: TensorFlow/Keras

## Output Comparison

### Original System Output:
```
Emotion: Happy (Confidence: 0.87)
```

### New System Output:
```
Expressiveness: Balanced Expression (Confidence: 0.82)
```

## Real-time Recognition

The system provides real-time facial expressiveness recognition via webcam, similar to the original system:

1. **Face Detection**: Uses MediaPipe to detect faces in real-time
2. **Expressiveness Classification**: Predicts expressiveness level
3. **Display**: Shows bounding box and prediction on video feed

## Files Structure

```
FYP/
├── Facial_Expression_Expressiveness_Recognition.ipynb  # Main notebook
├── real_time_demo.py                                   # Real-time demo script
├── analyze_facial_data.py                             # Data analysis script
├── quick_analysis.py                                  # Quick analysis script
├── requirements.txt                                   # Python dependencies
├── models/                                            # Saved models (after training)
│   ├── Facial_Expressiveness_Recognition_Model.h5
│   ├── Facial_Expressiveness_Recognition_Model.json
│   └── expressiveness_model_weights.h5
└── FYP/RecruitView_Data/                              # Dataset
    ├── metadata.jsonl                                 # Video metadata
    └── videos/                                        # Video files
```

## Training Process

1. **Data Analysis**: Analyze facial expression score distribution
2. **Face Extraction**: Extract faces from interview videos
3. **Model Training**: Train CNN on expressiveness categories
4. **Evaluation**: Assess model performance
5. **Real-time Testing**: Test with webcam

## Use Cases

- **Interview Analysis**: Assess candidate expressiveness during video interviews
- **Communication Research**: Study facial expressiveness patterns
- **HR Applications**: Evaluate non-verbal communication skills

## Advantages Over Original System

1. **Domain-Specific**: Trained on interview data instead of general faces
2. **Expressiveness Focus**: Measures communication style rather than emotions
3. **Modern Tech Stack**: Uses MediaPipe and TensorFlow instead of OpenCV/PyTorch
4. **Balanced Categories**: Uses statistical percentiles for fair category distribution

## Limitations

- Requires trained model before real-time use
- Limited to 3 expressiveness categories (vs 7 emotions)
- Trained specifically for interview contexts

## Troubleshooting

### Model Not Found
- Run the notebook first to train the model
- Check that model files exist in `models/` directory

### No Face Detected
- Ensure good lighting
- Position face clearly in camera view
- MediaPipe requires visible facial features

### Poor Recognition
- Model needs more training data
- Try different lighting conditions
- Ensure face is not obstructed

## Next Steps

1. Train the model using the notebook
2. Test real-time recognition
3. Fine-tune model parameters if needed
4. Integrate with interview analysis systems