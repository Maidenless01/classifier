import os
from pathlib import Path

import tensorflow as tf
from clean_data import clean_dataset
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Rescaling
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image_dataset_from_directory

print("TensorFlow version:", tf.__version__)


def configure_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    print("Num GPUs Available:", len(gpus))

    if not gpus:
        print("No GPU detected by TensorFlow -> running on CPU")
        return

    try:
        # Avoid pre-allocating all GPU memory at startup.
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        gpu_names = []
        for gpu in gpus:
            details = tf.config.experimental.get_device_details(gpu)
            gpu_names.append(details.get("device_name", gpu.name))
        print("GPU(s) found:", gpu_names)
    except RuntimeError as e:
        print("GPU setup warning:", e)


configure_gpu()

# Define paths
BASE_DIR = Path(__file__).resolve().parent
dataset_dir = BASE_DIR / "PetImages"
batch_size = 32
img_size = (160, 160)

# Remove corrupt/unreadable images before building TF datasets
print("Scanning for corrupt images...")
clean_dataset(str(dataset_dir))

print(f"Loading datasets from {dataset_dir}...")
if not os.path.exists(dataset_dir):
    print("Error: Dataset directory not found.")
    exit(1)

# Create training and validation datasets
train_dataset = image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

validation_dataset = image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

class_names = train_dataset.class_names
print(f"Class names found: {class_names}")

# Performance tuning
# shuffle before cache so different orderings are cached across epochs
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.shuffle(1000).cache().prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# Data augmentation
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
])

# Build a custom CNN model
inputs = tf.keras.Input(shape=img_size + (3,))
x = data_augmentation(inputs)
x = Rescaling(1./255)(x)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
# We have 2 classes (Cat and Dog).
# Binary classification: output one node with sigmoid activation.
# Alphabetically, Cat=0, Dog=1
outputs = Dense(1, activation='sigmoid')(x)
model = Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

print(model.summary())

# Train the model
epochs = 16 # Small number of epochs for quick demonstration
history = model.fit(train_dataset, epochs=epochs, validation_data=validation_dataset)

# Save the model
model_path = BASE_DIR / "cat_dog_model.keras"
model.save(model_path)
print(f"Model saved to {model_path}")
