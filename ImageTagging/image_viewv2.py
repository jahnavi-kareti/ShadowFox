import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import layers, models

# ──────────────────────────────────────────
# STEP 1: Load Dataset
# ──────────────────────────────────────────
print("TensorFlow Version:", tf.__version__)
print("Loading CIFAR-10...\n")

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# ──────────────────────────────────────────
# STEP 2: Normalize
# ──────────────────────────────────────────
x_train, x_test = x_train / 255.0, x_test / 255.0
print(f"Train: {x_train.shape} | Test: {x_test.shape}\n")

# ──────────────────────────────────────────
# STEP 3: Improved CNN Model
#   Changes from v1:
#   ✅ 3 Conv blocks instead of 2
#   ✅ BatchNormalization (stabilizes training)
#   ✅ Dropout (reduces overfitting)
#   ✅ More filters (32 → 64 → 128)
#   ✅ Larger Dense layer (64 → 256)
# ──────────────────────────────────────────
model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),

    # Block 1
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),                  # Drop 25% neurons

    # Block 2
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    # Block 3
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    # Classifier
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),                   # Drop 50% in Dense layer
    layers.Dense(10, activation='softmax')
])

# ──────────────────────────────────────────
# STEP 4: Compile with Learning Rate Scheduler
# ──────────────────────────────────────────
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ──────────────────────────────────────────
# STEP 5: Callbacks (auto reduce LR + early stop)
# ──────────────────────────────────────────
callbacks = [
    # Reduce learning rate when val_loss stops improving
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6
    ),
    # Stop early if no improvement for 7 epochs
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=7, restore_best_weights=True, verbose=1
    ),
    # Save best model automatically
    tf.keras.callbacks.ModelCheckpoint(
        'ImageTagging/best_model.keras', monitor='val_accuracy',
        save_best_only=True, verbose=1
    )
]

# ──────────────────────────────────────────
# STEP 6: Train
# ──────────────────────────────────────────
print("\nTraining started (improved model)...")
history = model.fit(
    x_train, y_train,
    epochs=30,                             # More epochs, early stop will kick in
    batch_size=64,
    validation_data=(x_test, y_test),
    callbacks=callbacks
)

# ──────────────────────────────────────────
# STEP 7: Evaluate
# ──────────────────────────────────────────
print("\nEvaluating...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\n✅ Final Test Accuracy : {test_acc * 100:.2f}%")
print(f"   Final Test Loss     : {test_loss:.4f}")
print(f"\n   (Previous model was: 69.55% — improvement: +{(test_acc*100 - 69.55):.2f}%)")

# ──────────────────────────────────────────
# STEP 8: Plot Training History
# ──────────────────────────────────────────
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'],     label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Val Accuracy',   marker='o')
plt.title('Model Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'],     label='Train Loss', marker='o', color='orange')
plt.plot(history.history['val_loss'], label='Val Loss',   marker='o', color='red')
plt.title('Model Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("ImageTagging/training_history_v2.png")
plt.show()
print("Saved: training_history_v2.png")

# ──────────────────────────────────────────
# STEP 9: Predict and Visualize
# ──────────────────────────────────────────
print("\nGenerating predictions...")

plt.figure(figsize=(14, 6))
for i in range(10):
    img = x_test[i]
    actual = class_names[y_test[i][0]]
    pred_probs = model.predict(np.expand_dims(img, axis=0), verbose=0)
    predicted = class_names[np.argmax(pred_probs)]
    confidence = np.max(pred_probs) * 100

    plt.subplot(2, 5, i + 1)
    plt.imshow(img)
    color = 'green' if predicted == actual else 'red'
    plt.title(f"P: {predicted}\nA: {actual}\n{confidence:.1f}%", color=color, fontsize=8)
    plt.axis('off')

plt.suptitle("Predictions  |  Green = Correct   Red = Wrong", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("ImageTagging/predictions_v2.png")
plt.show()
print("Saved: predictions_v2.png")

# ──────────────────────────────────────────
# STEP 10: Save Final Model
# ──────────────────────────────────────────
model.save("ImageTagging/cifar10_model_v2.keras")
print("\n✅ Model saved: cifar10_model_v2.keras")
print("\n🎉 Improved Model Complete!")