#ResNet
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import matplotlib.pyplot as plt
from google.colab import drive
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
import seaborn as sns

# Load dataset
drive.mount('/content/drive')
data = np.load('/content/drive/My Drive/content/MNIST_MIX_train_test.npz')
x_train, y_train = data['X_train'], data['y_train']
x_test, y_test = data['X_test'], data['y_test']

# Normalize and reshape data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Add channel dimension if needed (for grayscale)
if len(x_train.shape) == 3:
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

# Convert labels to one-hot encoding
num_classes = len(np.unique(y_train))
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)

# ResNet Block
def resnet_block(inputs, filters, kernel_size=3, stride=1, activation='relu'):
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same',
                      kernel_regularizer=regularizers.l2(0.001))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)

    x = layers.Conv2D(filters, kernel_size, padding='same',
                      kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)

    # Shortcut connection
    if inputs.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(inputs)
        shortcut = layers.BatchNormalization()(shortcut)
    else:
        shortcut = inputs

    x = layers.Add()([x, shortcut])
    x = layers.Activation(activation)(x)
    return x

# Build ResNet Model
def build_resnet(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)

    # Initial conv layer
    x = layers.Conv2D(32, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # ResNet blocks
    x = resnet_block(x, 32)
    x = layers.MaxPooling2D(2)(x)

    x = resnet_block(x, 64)
    x = layers.MaxPooling2D(2)(x)

    x = resnet_block(x, 128)
    x = layers.GlobalAveragePooling2D()(x)

    # Dense layers
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs, outputs)

# Create model
model = build_resnet(x_train.shape[1:], num_classes)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Add callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)
]

# Train model
history = model.fit(x_train, y_train_cat,
                   batch_size=64,
                   epochs=50,
                   validation_data=(x_test, y_test_cat),
                   callbacks=callbacks,
                   verbose=1)

# Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
print(f'\nTest accuracy: {test_acc*100:.2f}%')

# Plot accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')

# Get predictions for the test set
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes))

print(f"\nF1 Score (macro): {f1_score(y_test, y_pred_classes, average='macro'):.4f}")
print(f"Precision (macro): {precision_score(y_test, y_pred_classes, average='macro'):.4f}")
print(f"Recall (macro): {recall_score(y_test, y_pred_classes, average='macro'):.4f}")

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
plt.subplot(1, 2, 2)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()