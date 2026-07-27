"""
Program 13: Image Classification using Keras (MNIST Digits)
Time Complexity: O(epochs * n * L * w) for training
Space Complexity: O(L * w) for model parameters
Requires: pip install tensorflow
"""

import numpy as np


def build_and_train():
    """Build and train a CNN for MNIST digit classification."""
    from tensorflow import keras
    from tensorflow.keras import layers

    # Load MNIST dataset
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    # Preprocess: normalize and reshape
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    # Convert labels to one-hot encoding
    y_train_cat = keras.utils.to_categorical(y_train, 10)
    y_test_cat = keras.utils.to_categorical(y_test, 10)

    # Build CNN model
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu',
                      input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Model Summary:")
    model.summary()

    # Train the model
    print("\nTraining...")
    history = model.fit(
        X_train, y_train_cat,
        epochs=5,
        batch_size=128,
        validation_split=0.1,
        verbose=1
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\nTest Accuracy: {test_acc * 100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    # Predict on sample
    sample_indices = [0, 1, 2, 3, 4]
    predictions = model.predict(X_test[sample_indices], verbose=0)

    print("\nSample Predictions:")
    for i, idx in enumerate(sample_indices):
        pred_label = np.argmax(predictions[i])
        confidence = predictions[i][pred_label] * 100
        actual = y_test[idx]
        status = "✓" if pred_label == actual else "✗"
        print(f"  Sample {idx}: Predicted={pred_label} "
              f"(confidence: {confidence:.1f}%) Actual={actual} {status}")

    return model


# Driver Code
if __name__ == "__main__":
    print("=== Image Classification using Keras (MNIST) ===\n")
    try:
        model = build_and_train()
    except ImportError:
        print("TensorFlow/Keras not installed.")
        print("Install with: pip install tensorflow")
        print("\nThis program builds a CNN with architecture:")
        print("  Conv2D(32) -> MaxPool -> Conv2D(64) -> MaxPool ->")
        print("  Flatten -> Dense(64) -> Dropout(0.5) -> Dense(10)")
        print("  Expected accuracy: ~99% on MNIST test set")
