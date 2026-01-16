# Facial Expression Expressiveness Recognition System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17+-orange.svg)](https://tensorflow.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-green.svg)](https://mediapipe.dev/)

A novel facial expression recognition system that classifies **expressiveness levels** rather than traditional emotions. Perfect for interview analysis and communication research.

## 🎯 What's Different

| Feature | Traditional Emotion Recognition | This System |
|---------|-------------------------------|-------------|
| **Categories** | 7 emotions (anger, fear, happy, etc.) | **3 expressiveness levels** |
| **Face Detection** | OpenCV Haar cascades | **MediaPipe** (Google's framework) |
| **Deep Learning** | PyTorch + TorchVision | **TensorFlow/Keras** |
| **Dataset** | FER2013 (general faces) | **RecruitView_Data** (interview videos) |
| **Use Case** | General emotion detection | **Interview expressiveness analysis** |

## 🚀 Quick Start

### 1. Clone & Setup Environment
```bash
# Clone repository
git clone https://github.com/yourusername/facial-expression-expressiveness.git
cd facial-expression-expressiveness

# Create virtual environment
python -m venv facial_expressiveness_env
facial_expressiveness_env\Scripts\activate  # Windows
# source facial_expressiveness_env/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Test the System
```bash
# Test if everything works
python test_system.py
```

### 3. Train Your Model
```bash
# Open Jupyter notebook
jupyter notebook Facial_Expression_Expressiveness_Recognition.ipynb

# Run all cells to train the model
```

### 4. Run Real-Time Recognition
```bash
# Start webcam recognition
python real_time_demo.py
```

## 📊 Expressiveness Categories

The system classifies facial expressions into three levels based on statistical analysis:

- **😐 Reserved Expression**: Low facial expressiveness (score < -0.303)
- **🙂 Balanced Expression**: Neutral expressiveness (-0.303 ≤ score ≤ 0.294)
- **😊 Expressive**: High expressiveness (score > 0.294)

## 🏗️ Project Structure

```
facial-expression-expressiveness/
├── 📓 Facial_Expression_Expressiveness_Recognition.ipynb  # Main training notebook
├── 🎥 real_time_demo.py                                   # Real-time webcam demo
├── 🧪 test_system.py                                      # System testing
├── 📋 requirements.txt                                    # Python dependencies
├── 📖 README.md                                            # This file
├── 📚 USAGE_GUIDE.md                                       # Detailed documentation
├── 🔍 analyze_facial_data.py                              # Data analysis tools
├── 📊 quick_analysis.py                                   # Fast data analysis
└── 📁 FYP/RecruitView_Data/                               # Interview dataset
    ├── 📄 metadata.jsonl                                  # Interview metadata
    └── 🎬 videos/                                         # Interview video files
```

## 🎥 Real-Time Demo

The system provides live facial expressiveness recognition via webcam:

```bash
python real_time_demo.py
```

**Features:**
- ✅ Real-time face detection using MediaPipe
- ✅ Live expressiveness classification
- ✅ Visual feedback with bounding boxes
- ✅ Confidence scores display
- ✅ Press 'q' to quit

**Sample Output:**
```
Balanced Expression: 0.87
```

## 🤖 Training Process

### Data Pipeline
1. **Load Interview Data**: Parse metadata from 2011 interview records
2. **Face Extraction**: Extract faces from video frames using MediaPipe
3. **Category Classification**: Label faces based on expressiveness scores
4. **Data Augmentation**: Apply rotations, shifts, and flips for robustness

### Model Architecture
```
Input (48x48 grayscale)
    ↓
Conv2D (32 filters) → BatchNorm → ReLU → MaxPool → Dropout (0.25)
    ↓
Conv2D (64 filters) → BatchNorm → ReLU → MaxPool → Dropout (0.25)
    ↓
Conv2D (128 filters) → BatchNorm → ReLU → MaxPool → Dropout (0.25)
    ↓
Flatten → Dense (256) → BatchNorm → Dropout (0.5) → Dense (3) → Softmax
```

### Training Configuration
- **Optimizer**: Adam (lr=0.001)
- **Loss**: Categorical Crossentropy
- **Batch Size**: 32
- **Epochs**: Up to 50 (with early stopping)
- **Data Augmentation**: Rotation, shift, flip, zoom

## 📈 Performance

**Expected Results:**
- **Training Time**: 10-30 minutes
- **Test Accuracy**: 75-85%
- **Real-time FPS**: 15-30 FPS (CPU)

## 🔧 Requirements

- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB for models and data
- **Camera**: Webcam for real-time demo
- **GPU**: Optional (TensorFlow uses CPU by default)

## 📦 Dependencies

```
tensorflow==2.16.1          # Deep learning framework
keras==3.0.5                # High-level neural networks API
mediapipe==0.10.11          # Face detection and tracking
opencv-python==4.9.0.80     # Computer vision
numpy==1.26.4               # Numerical computing
pandas==2.2.2               # Data manipulation
matplotlib==3.8.4           # Plotting
seaborn==0.13.2             # Statistical visualization
scikit-learn==1.4.2         # Machine learning utilities
tqdm==4.66.4                # Progress bars
jupyter==1.0.0              # Interactive notebooks
```

## 🎯 Use Cases

- **🎤 Interview Analysis**: Evaluate candidate expressiveness during video interviews
- **📚 Communication Research**: Study facial expressiveness patterns
- **🏢 HR Applications**: Assess non-verbal communication skills
- **🤝 Behavioral Analysis**: Understand communication styles

## 🔍 Key Advantages

✅ **Domain-Specific**: Trained on interview data instead of general faces
✅ **Expressiveness Focus**: Measures communication style rather than basic emotions
✅ **Modern Tech Stack**: Uses cutting-edge MediaPipe and TensorFlow
✅ **Balanced Categories**: Uses statistical percentiles for fair distribution
✅ **No Overlap**: Completely different approach from traditional emotion recognition

## 🚨 Troubleshooting

### Model Not Found
```bash
# Train the model first
jupyter notebook Facial_Expression_Expressiveness_Recognition.ipynb
# Run all cells to create models/ directory
```

### Import Errors
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Webcam Issues
```bash
# Check camera permissions
# Close other camera applications
python test_system.py  # Test camera access
```

### Memory Issues
```bash
# Reduce batch size in notebook
# Process fewer videos during training
# Close other applications
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **RecruitView_Data**: Interview dataset for training
- **MediaPipe**: Google's face detection framework
- **TensorFlow**: Deep learning framework
- **Original Inspiration**: live-face-emotion-classifier project

## 📞 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Ready to analyze facial expressiveness?** 🚀

```bash
python real_time_demo.py
```