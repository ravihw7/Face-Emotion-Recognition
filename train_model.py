import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'        # Suppress INFO & WARNING logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'        # Enable oneDNN for AVX/AVX2/FMA acceleration

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

# Paths
train_dir = "archive/train"
test_dir = "archive/test"

# Image parameters
img_size = 48
batch_size = 64

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.15,
    shear_range=0.1,
    brightness_range=[0.8, 1.2],
)

# Only rescale for validation/test
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_size, img_size),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False,
)

# Print class mapping
print("Class indices:", train_generator.class_indices)

# ---- Compute class weights to handle imbalance ----
class_labels = train_generator.classes
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(class_labels),
    y=class_labels,
)
class_weight_dict = dict(enumerate(class_weights))
print("Class weights:", class_weight_dict)

# ---- Deeper CNN Model ----
model = Sequential()

# Block 1
model.add(Conv2D(64, (3, 3), padding="same", activation="relu", input_shape=(img_size, img_size, 1)))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Block 2
model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Block 3
model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Block 4
model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Fully connected
model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(256, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(7, activation="softmax"))

# Compile with lower learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# Callbacks
early_stop = EarlyStopping(
    monitor="val_accuracy", patience=10, restore_best_weights=True, mode="max"
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7, verbose=1
)

# Train with class weights
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=test_generator,
    class_weight=class_weight_dict,
    callbacks=[early_stop, reduce_lr],
)

# Save model
os.makedirs("models", exist_ok=True)
model.save("models/emotion_model.h5")

print("\nModel training complete and saved.")
print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")