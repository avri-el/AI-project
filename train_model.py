import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
IMAGE_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
CLASS_LABELS = [
    "Colon Adenocarcinoma",
    "Colon Benign Tissue",
    "Colon Malignant",
    "Normal Colon Tissue"
]

def create_model(num_classes):
    """Create a CNN model for colon disease classification"""
    model = Sequential([
        # First conv block
        Conv2D(32, 3, activation='relu', padding='same', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        BatchNormalization(),
        Conv2D(32, 3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2),
        Dropout(0.25),
        
        # Second conv block
        Conv2D(64, 3, activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, 3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2),
        Dropout(0.25),
        
        # Third conv block
        Conv2D(128, 3, activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, 3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2),
        Dropout(0.25),
        
        # Dense layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

def plot_training_history(history):
    """Plot accuracy and loss curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Training')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Loss
    ax2.plot(history.history['loss'], label='Training')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def main():
    # Setup data generators with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    # Load training / validation data
    print("Loading training data...")
    # Support two common layouts:
    # 1) dataset/train/<class> (uses validation_split)
    # 2) dataset/train/{train,val,test}/<class> (separate folders)
    DATA_ROOT = 'dataset/train'
    train_dir = DATA_ROOT
    validation_dir = None
    test_dir = None

    if os.path.exists(os.path.join(DATA_ROOT, 'train')):
        # Layout (2)
        train_dir = os.path.join(DATA_ROOT, 'train')
        validation_dir = os.path.join(DATA_ROOT, 'val') if os.path.exists(os.path.join(DATA_ROOT, 'val')) else None
        test_dir = os.path.join(DATA_ROOT, 'test') if os.path.exists(os.path.join(DATA_ROOT, 'test')) else None

        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(IMAGE_SIZE, IMAGE_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )

        if validation_dir:
            validation_generator = train_datagen.flow_from_directory(
                validation_dir,
                target_size=(IMAGE_SIZE, IMAGE_SIZE),
                batch_size=BATCH_SIZE,
                class_mode='categorical'
            )
        else:
            validation_generator = None

        test_generator = None
        if test_dir:
            test_generator = test_datagen.flow_from_directory(
                test_dir,
                target_size=(IMAGE_SIZE, IMAGE_SIZE),
                batch_size=BATCH_SIZE,
                class_mode='categorical',
                shuffle=False
            )
    else:
        # Layout (1): use validation_split on DATA_ROOT
        train_generator = train_datagen.flow_from_directory(
            DATA_ROOT,
            target_size=(IMAGE_SIZE, IMAGE_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='training'
        )

        validation_generator = train_datagen.flow_from_directory(
            DATA_ROOT,
            target_size=(IMAGE_SIZE, IMAGE_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='validation'
        )

        test_generator = None
        if os.path.exists('dataset/test'):
            test_generator = test_datagen.flow_from_directory(
                'dataset/test',
                target_size=(IMAGE_SIZE, IMAGE_SIZE),
                batch_size=BATCH_SIZE,
                class_mode='categorical',
                shuffle=False
            )

    # Create and compile model
    print("Creating model...")
    model = create_model(len(CLASS_LABELS))
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            'model/colon_diseases.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Train model
    print("Training model...")
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Plot training history
    plot_training_history(history)

    # Evaluate on test set if available
    if test_generator:
        print("\nEvaluating on test set...")
        test_loss, test_acc = model.evaluate(test_generator)
        print(f"Test accuracy: {test_acc:.4f}")
        print(f"Test loss: {test_loss:.4f}")

        # Generate predictions
        predictions = model.predict(test_generator)
        y_pred = np.argmax(predictions, axis=1)
        y_true = test_generator.classes

        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=CLASS_LABELS))

        # Plot confusion matrix
        plot_confusion_matrix(y_true, y_pred, CLASS_LABELS)

if __name__ == '__main__':
    main()