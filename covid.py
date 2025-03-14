import numpy as np
import pandas as pd
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns

# ========================
# 1. Dataset Preparation
# ========================
train_dir = '/content/drive/MyDrive/projet_AI/DATA/train'
val_dir = '/content/drive/MyDrive/projet_AI/DATA/val'
test_dir = '/content/drive/MyDrive/projet_AI/DATA/test'

# Data Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Training Data Generator
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(227, 227),
    batch_size=32,
    class_mode='binary'
)

# Validation Data Generator
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(227, 227),
    batch_size=32,
    class_mode='binary'
)

# Test Data Generator

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(227, 227),
    batch_size=1,
    class_mode='binary',
    shuffle=False
)

# ========================
# 2. Model Architecture
# ========================
# Using VGG16 as a pre-trained model for transfer learning
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(227, 227, 3))
base_model.trainable = False  # Freeze the base model

model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Learning Rate Scheduler
lr_schedule = ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Compile the model
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# ========================
# 3. Callbacks
# ========================
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# ========================
# 4. Training the Model
# ========================
# Class Weights to Handle Imbalance
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    class_weight=dict(enumerate(class_weights)),
    callbacks=[early_stopping]
)

model.summary()

# ========================
# 5. Evaluating the Model
# ========================
# Evaluate on the test set
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.2f}")


# Extracting data for visualization
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
epochs_range = range(1, len(train_accuracy) + 1)

# Plot the curves
plt.figure(figsize=(8, 6))
plt.plot(epochs_range, train_accuracy, label='Training Accuracy')
plt.plot(epochs_range, val_accuracy, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)

# ========================
# 6. Curves & Visualizing
# ========================
# Confusion Matrix
y_pred = model.predict(test_generator)
y_pred_classes = (y_pred > 0.5).astype(int).flatten()
y_true = test_generator.classes

# Confusion Matrix Visualization
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'COVID-19'], yticklabels=['Normal', 'COVID-19'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes))


# Compute ROC Curve
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})", color='blue')
plt.plot([0, 1], [0, 1], 'r--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.grid(True)
plt.show()

# ========================
# 7. Predicting on New Image
# ========================
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Example image path
img_path = '/content/drive/MyDrive/projet_AI/DATA/test/normal/Normal-1383.png'

# Load and preprocess the image
img = image.load_img(img_path, target_size=(227, 227))
plt.imshow(img)
plt.show()

img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Predict
score = model.predict(img_array)
prediction = 'COVID' if score < 0.5 else 'Normal'
print(f"Predicted: {prediction}, Score: {score[0][0]:.4f}")
