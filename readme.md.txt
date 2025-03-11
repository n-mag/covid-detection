# COVID-19 Detection using Deep Learning

This project utilizes a deep learning model based on transfer learning with VGG16 to classify chest X-ray images as either COVID-19 positive or normal. The model is trained using TensorFlow and Keras, with dataset augmentation and performance evaluation techniques.

## Table of Contents
- [Dataset](#dataset)
- [Installation](#installation)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Prediction on New Images](#prediction-on-new-images)
- [Results](#results)
- [License](#license)

## Dataset
The dataset is structured into three directories:
- `train/` - Training images
- `val/` - Validation images
- `test/` - Test images

Each directory contains two subfolders:
- `COVID-19/` - Images labeled as COVID-19 positive
- `Normal/` - Images labeled as normal

## Installation
To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/covid-detection.git
   cd covid-detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure your dataset is placed in the correct directory.

## Model Architecture
The model is built using the VGG16 architecture for feature extraction, followed by a custom classification head:
- **VGG16 as the base model (pre-trained on ImageNet, frozen layers)**
- **Flatten layer**
- **Fully connected layer with 256 neurons and ReLU activation**
- **Dropout layer (0.5 probability)**
- **Output layer with sigmoid activation (binary classification)**

## Training
To train the model, execute:
```python
python covid.py
```
- **Data Augmentation** is applied to prevent overfitting.
- **Class Weights** are computed to handle data imbalance.
- **Early Stopping** is used to optimize training.

## Evaluation
After training, the model is evaluated using:
- **Accuracy on test data**
- **Confusion Matrix**
- **Classification Report (Precision, Recall, F1-score)**
- **ROC Curve & AUC score**

## Prediction on New Images
To predict on a new chest X-ray image, update the `img_path` variable in `covid.py` and run the script. The model will output whether the image is classified as COVID-19 or normal.

## Results
The model performance is visualized using:
- **Accuracy curves**
- **Loss curves**
- **ROC curve for classification performance**

## License
This project is open-source and available under the MIT License.

