# Chest X-ray Pneumonia Detection with Grad-CAM

A deep learning application for detecting pneumonia in chest X-ray images with explainable AI visualization.

## Features

- **ResNet50 Transfer Learning**: High-accuracy pneumonia detection
- **Grad-CAM Visualization**: Explainable AI showing model decision regions
- **Streamlit Web Interface**: Easy-to-use interactive application
- **Comprehensive Evaluation**: Confusion matrix, ROC curves, and classification metrics
- **Data Augmentation**: Robust training with image transformations

## Project Structure

\`\`\`
.
├── scripts/
│   ├── data_processing.py      # Data loading and preprocessing
│   ├── gradcam.py              # Grad-CAM implementation
│   ├── train_model.py          # Model training script
│   ├── evaluate_model.py       # Model evaluation metrics
│   └── app.py                  # Streamlit web application
├── requirements.txt            # Python dependencies
└── README.md                   # This file
\`\`\`

## Installation

1. Clone the repository
2. Install dependencies:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

## Usage

### Training the Model

1. Prepare your dataset with structure:
   \`\`\`
   data/
   ├── normal/
   │   ├── image1.jpg
   │   └── ...
   └── pneumonia/
       ├── image1.jpg
       └── ...
   \`\`\`

2. Update paths in `train_model.py` and run:
   \`\`\`bash
   python scripts/train_model.py
   \`\`\`

### Running the Web Application

\`\`\`bash
streamlit run scripts/app.py
\`\`\`

Then open your browser to `http://localhost:8501`

### Model Evaluation

\`\`\`python
from evaluate_model import ModelEvaluator
import tensorflow as tf

model = tf.keras.models.load_model('pneumonia_detection_model.h5')
evaluator = ModelEvaluator(model)
results = evaluator.generate_full_report(X_test, y_test)
\`\`\`

## Model Architecture

- **Base Model**: ResNet50 (pretrained on ImageNet)
- **Input**: 224×224 RGB images
- **Custom Head**:
  - Global Average Pooling
  - Dense(256, relu) + Dropout(0.3)
  - Dense(128, relu) + Dropout(0.2)
  - Dense(1, sigmoid) - Binary classification

## Grad-CAM Explanation

Grad-CAM (Gradient-weighted Class Activation Mapping) visualizes which regions of the input image are important for the model's prediction. The heatmap overlay shows:
- **Red/Yellow regions**: High importance for prediction
- **Blue regions**: Low importance

## Performance Metrics

The model is evaluated using:
- Accuracy
- Precision & Recall
- F1-Score
- Confusion Matrix
- ROC-AUC Score

## Disclaimer

This application is for educational and research purposes only. It should not be used for clinical diagnosis without professional medical review.

## License

MIT License
