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
def prepare_data(train_dir, val_dir, test_dir):
    """
    Prepares the dataset by applying data augmentation and rescaling.

    Args:
        train_dir (str): Path to the training dataset.
        val_dir (str): Path to the validation dataset.
        test_dir (str): Path to the test dataset.

    Returns:
        tuple: ImageDataGenerator objects for training, validation, and testing.
    """
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

    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(227, 227),
        batch_size=32,
        class_mode='binary'
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(227, 227),
        batch_size=32,
        class_mode='binary'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(227, 227),
        batch_size=1,
        class_mode='binary',
        shuffle=False
    )

    return train_generator, val_generator, test_generator

# ========================
# 2. Model Architecture
# ========================

def build_model():
   
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(227, 227, 3))
    base_model.trainable = False  # Freeze the base model

    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification output
    ])

    lr_schedule = ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=1000,
        decay_rate=0.96,
        staircase=True
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model

# ========================
# 3. Training the Model
# ========================
def train_model(model, train_generator, val_generator):
    """
    Trains the CNN model using the given dataset.

    Args:
        model (keras.Model): Compiled CNN model.
        train_generator (ImageDataGenerator): Training dataset generator.
        val_generator (ImageDataGenerator): Validation dataset generator.

    Returns:
        keras.callbacks.History: Training history.
    """
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=50,
        class_weight=dict(enumerate(class_weights)),
        callbacks=[early_stopping]
    )

    return history


# ========================
# 6. Evaluating the Model
# ========================
def evaluate_model(model, test_generator, history):
    """
    Evaluates the trained model on the test dataset and plots performance metrics.

    Args:
        model (keras.Model): Trained CNN model.
        test_generator (ImageDataGenerator): Test dataset generator.
        history (keras.callbacks.History): Training history.

    Returns:
        float: Test accuracy.
    """
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Test Accuracy: {test_acc:.2f}")

    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    epochs_range = range(1, len(train_accuracy) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, train_accuracy, label='Training Accuracy')
    plt.plot(epochs_range, val_accuracy, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    return test_acc

# ========================
# 7. Curves & Visualizing
# ========================
def visualize_performance(model, test_generator):
    """
    Generates and displays confusion matrix and ROC curve.

    Args:
        model (keras.Model): Trained CNN model.
        test_generator (ImageDataGenerator): Test dataset generator.
    """
    y_pred = model.predict(test_generator)
    y_pred_classes = (y_pred > 0.5).astype(int).flatten()
    y_true = test_generator.classes

    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'COVID-19'], yticklabels=['Normal', 'COVID-19'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes))

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

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
# 8. Predicting on New Image
# ========================
def predict_image(model, img_path):
    
    from tensorflow.keras.preprocessing import image

    img = image.load_img(img_path, target_size=(227, 227))
    plt.imshow(img)
    plt.show()

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    score = model.predict(img_array)
    prediction = 'COVID' if score < 0.5 else 'Normal'
    print(f"Predicted: {prediction}, Score: {score[0][0]:.4f}")
    return prediction
    
# ========================
# Run the Pipeline
# ========================
if __name__ == "__main__":
    train_dir = '/content/drive/MyDrive/projet_AI/DATA/train'
    val_dir = '/content/drive/MyDrive/projet_AI/DATA/val'
    test_dir = '/content/drive/MyDrive/projet_AI/DATA/test'
    img_path = '/content/drive/MyDrive/projet_AI/DATA/test/normal/Normal-1383.png'

    train_generator, val_generator, test_generator = prepare_data(train_dir, val_dir, test_dir)
    model = build_model()
    history = train_model(model, train_generator, val_generator)
    evaluate_model(model, test_generator, history)
    visualize_performance(model, test_generator)
    predict_image(model, img_path)
