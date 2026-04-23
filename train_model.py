import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# Configuration
N_MFCC = 13
MAX_TIME_STEPS = 100
BATCH_SIZE = 32
EPOCHS = 50
MODEL_SAVE_PATH = "models/wake_word_model.h5"


def build_dscnn_model(input_shape, num_classes=2):
    """
    Build Depthwise Separable CNN (DS-CNN) model.
    
    DS-CNN reduces computational complexity by separating convolution
    into depthwise and pointwise operations — ideal for embedded systems.
    
    Parameters:
        input_shape : (time_steps, n_mfcc, 1)
        num_classes : 2 (wake word / background)
    
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([

        # Input layer
        layers.Input(shape=input_shape),

        # Standard Conv layer (first layer)
        layers.Conv2D(32, (3, 3), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),

        # DS-CNN Block 1
        layers.DepthwiseConv2D((3, 3), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2D(32, (1, 1), use_bias=False),   # Pointwise
        layers.BatchNormalization(),
        layers.ReLU(),

        # DS-CNN Block 2
        layers.DepthwiseConv2D((3, 3), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2D(64, (1, 1), use_bias=False),   # Pointwise
        layers.BatchNormalization(),
        layers.ReLU(),

        # DS-CNN Block 3
        layers.DepthwiseConv2D((3, 3), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2D(64, (1, 1), use_bias=False),   # Pointwise
        layers.BatchNormalization(),
        layers.ReLU(),

        # Global Average Pooling
        layers.GlobalAveragePooling2D(),

        # Dropout for regularization
        layers.Dropout(0.3),

        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_model():
    """Load data, build model, train and save."""

    # Load dataset
    print("Loading dataset...")
    X = np.load("X_data.npy")
    y = np.load("y_data.npy")

    # Reshape for CNN input: (samples, time_steps, n_mfcc, 1)
    X = X[..., np.newaxis]

    print(f"Dataset shape: {X.shape}")
    print(f"Labels: {np.unique(y, return_counts=True)}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Build model
    input_shape = (MAX_TIME_STEPS, N_MFCC, 1)
    model = build_dscnn_model(input_shape)
    model.summary()

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True
        )
    ]

    # Train
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {test_acc * 100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    # Save final model
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    return model, history


if __name__ == "__main__":
    import os
    os.makedirs("models", exist_ok=True)
    model, history = train_model()
