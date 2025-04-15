# Image Splicing and Tampering Detection

This project implements an image tamper detection system using a Siamese Neural Network. It detects and highlights tampered regions between image pairs and classifies whether an image has been edited. The pipeline includes data preparation, training, visualization, and prediction functionalities.

---

## Features

- **Siamese network model** using TensorFlow and Keras
- Automatic creation of positive and negative image pairs
- Visual highlighting of differences in tampered image regions
- Model evaluation with:
  - Accuracy, Precision, Recall, F1-score
  - Confusion Matrix and ROC Curve
- Training history and evaluation visualizations
- Save/load trained models for reuse

---

## Model Architecture

- CNN-based base network with 3 convolutional layers
- `Custom DistanceLayer` computes absolute difference between embeddings
- Final sigmoid layer outputs similarity score:
  - `0 = same`, `1 = tampered`

---

## Requirements

### Install the required dependencies:

```bash
pip install -r requirements.txt

Required Libraries:
TensorFlow
OpenCV
NumPy
scikit-learn
matplotlib
seaborn
```
#### Preparing the Data
Place your images in the following structure:
original/     →  Original images (e.g., 1.jpg, 2.jpg, ...)
edited/       →  Tampered versions (e.g., 1.jpg, 2.jpg, ...)

Training the Model
Run the script to start training:

```bash
python your_script.py
```
This will:

Prepare training pairs
Train the Siamese model
Save training metrics and the trained model

### Outputs
training_history.png – Accuracy and loss plots
model_metrics.png – Evaluation visualizations
image_comparison_model.keras – Trained model
Console Output – Full classification report

### Comparing Image Pairs
Use the compare_all_pairs(model) function to detect tampered regions:
from your_script import train_model, compare_all_pairs

model = train_model()
compare_all_pairs(model)
Compares each image pair from original/ and edited/
Highlights differences using contours
Saves the output with bounding boxes
