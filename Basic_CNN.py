#Simple CNN
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import callbacks
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Load dataset
file_path = "/content/drive/My Drive/content/MNIST_MIX_train_test.npz"
data = np.load(file_path)
X_train, y_train = data['X_train'], data['y_train']
X_test, y_test = data['X_test'], data['y_test']

# Normalize and reshape
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Convert labels to one-hot with 100 classes
num_classes = 100
y_train_onehot = to_categorical(y_train, num_classes)
y_test_onehot = to_categorical(y_test, num_classes)

# Create simple model
model = Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train_onehot,
                    epochs=50,
                    batch_size=64,
                    validation_data=(X_test, y_test_onehot),
                    callbacks=[
                        callbacks.EarlyStopping(patience=5, restore_best_weights=True)
                    ])

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test_onehot)
print(f'\nTest accuracy: {test_acc*100:.2f}%')

# Get predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes))

# Calculate precision, recall, and F1 scores
precision_macro = precision_score(y_test, y_pred_classes, average='macro')
recall_macro = recall_score(y_test, y_pred_classes, average='macro')
f1_macro = f1_score(y_test, y_pred_classes, average='macro')

precision_micro = precision_score(y_test, y_pred_classes, average='micro')
recall_micro = recall_score(y_test, y_pred_classes, average='micro')
f1_micro = f1_score(y_test, y_pred_classes, average='micro')

precision_weighted = precision_score(y_test, y_pred_classes, average='weighted')
recall_weighted = recall_score(y_test, y_pred_classes, average='weighted')
f1_weighted = f1_score(y_test, y_pred_classes, average='weighted')

# Print metrics
print("\nAdditional Metrics:")
print(f"Macro Precision: {precision_macro:.4f}")
print(f"Macro Recall: {recall_macro:.4f}")
print(f"Macro F1 Score: {f1_macro:.4f}\n")

print(f"Micro Precision: {precision_micro:.4f}")
print(f"Micro Recall: {recall_micro:.4f}")
print(f"Micro F1 Score: {f1_micro:.4f}\n")

print(f"Weighted Precision: {precision_weighted:.4f}")
print(f"Weighted Recall: {recall_weighted:.4f}")
print(f"Weighted F1 Score: {f1_weighted:.4f}")

# Confusion matrix
plt.figure(figsize=(12, 10))
conf_mat = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(conf_mat, annot=False, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Plot training history
plt.figure(figsize=(12, 5))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.grid()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()