import numpy as np
import os
import glob
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --- Configuration ---
PATCH_SIZE = 7
IMAGE_DIR = r"D:\College\Btech Project\FINAL\attacks\prototype23\training_images_grayscale"
MODEL_SAVE_PATH = 'cnn_pixel_predictor.h5'
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42
BATCH_SIZE = 128
BUFFER_SIZE = 10000
STEPS_PER_EPOCH = 1000      # Cap how much training is done per epoch
VALIDATION_STEPS = 200      # Cap validation batches per epoch
EPOCHS = 50

def get_patches_from_image_normalized(image_np_normalized, patch_size):
    half_patch = patch_size // 2
    rows, cols = image_np_normalized.shape
    padded_img = np.pad(image_np_normalized, half_patch, mode='edge')
    for r in range(rows):
        for c in range(cols):
            patch = padded_img[r : r + patch_size, c : c + patch_size]
            label = image_np_normalized[r, c]
            yield patch.reshape(patch_size, patch_size, 1), label

def image_patch_generator(image_file_list, patch_size):
    while True:  # Infinite generator loop for tf.data
        for image_path in image_file_list:
            try:
                img = Image.open(image_path).convert('L')
                img_np = np.array(img, dtype=np.float32) / 255.0
                yield from get_patches_from_image_normalized(img_np, patch_size)
            except Exception as e:
                print(f"Skipping {image_path} due to error: {e}")
                continue

# --- Data Preparation ---
print("Scanning for training images...")
image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff')
all_image_files = []
for ext in image_extensions:
    all_image_files.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))

if not all_image_files:
    print(f"ERROR: No images found in '{IMAGE_DIR}'.")
    exit()

print(f"Found {len(all_image_files)} images.")

train_files, val_files = train_test_split(all_image_files, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED)
print(f"Training with {len(train_files)} images, validating with {len(val_files)} images.")

output_signature = (
    tf.TensorSpec(shape=(PATCH_SIZE, PATCH_SIZE, 1), dtype=tf.float32),
    tf.TensorSpec(shape=(), dtype=tf.float32)
)

print("Setting up training dataset...")
train_dataset = tf.data.Dataset.from_generator(
    lambda: image_patch_generator(train_files, PATCH_SIZE),
    output_signature=output_signature
)
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print("Setting up validation dataset...")
val_dataset = tf.data.Dataset.from_generator(
    lambda: image_patch_generator(val_files, PATCH_SIZE),
    output_signature=output_signature
)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# --- Model Definition ---
model = Sequential([
    Input(shape=(PATCH_SIZE, PATCH_SIZE, 1)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# --- Training ---
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True),
    ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True, verbose=1)
]

print("Starting training...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VALIDATION_STEPS,
    callbacks=callbacks,
    verbose=1
)

print(f"Training complete. Best model saved to {MODEL_SAVE_PATH}")
