import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
# STEP 1: Load Dataset
print("TensorFlow Version:", tf.__version__)
print("AI Environment Ready!\n")

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print("Training samples:", len(x_train))
print("Test samples    :", len(x_test))
# STEP 2: Normalize pixel values (0-255 → 0-1)
x_train, x_test = x_train / 255.0, x_test / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

print("Data normalized and ready!\n")
# STEP 3: Build CNN Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(32, 32, 3)),          # Proper Input layer (no warning)
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')    # 10 classes output
])
# STEP 4: Compile Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
# STEP 5: Train Model
print("\nTraining started...")
history = model.fit(
    x_train, y_train,
    epochs=10,
    validation_data=(x_test, y_test)
)
# STEP 6: Evaluate Model
print("\nEvaluating model...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\n✅ Final Test Accuracy: {test_acc * 100:.2f}%")
print(f"   Final Test Loss    : {test_loss:.4f}")
# STEP 7: Plot Training History
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'],    label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'],label='Val Accuracy',   marker='o')
plt.title('Model Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'],    label='Train Loss', marker='o', color='orange')
plt.plot(history.history['val_loss'],label='Val Loss',   marker='o', color='red')
plt.title('Model Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("ImageTagging/training_history.png")
plt.show()
print("Saved: training_history.png")
# STEP 8: Predict and Visualize Results
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
plt.savefig("ImageTagging/predictions.png")
plt.show()
print("Saved: predictions.png")
# STEP 9: Save the Model
model.save("ImageTagging/cifar10_model.keras")
print("\n✅ Model saved as: ImageTagging/cifar10_model.keras")
print("\n Project Complete!")