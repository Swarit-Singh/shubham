import tensorflow as tf
from tensorflow.keras import layers, Model
from ml_training.data_loader import load_region_dataset

# 1. Load data
X_train, y_train, X_val, y_val = load_region_dataset()

# 2. Build model
inp = layers.Input(shape=(32,32,1))
x = layers.Conv2D(16,3,activation='relu',padding='same')(inp)
x = layers.MaxPool2D()(x)
x = layers.Conv2D(32,3,activation='relu',padding='same')(x)
x = layers.MaxPool2D()(x)
x = layers.Flatten()(x)
out = layers.Dense(1, activation='sigmoid')(x)
model = Model(inp, out)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 3. Train
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5, batch_size=64
)

# 4. Save
model.save('models/ml_assisted/region_classifier.keras')
print("Saved region classifier to models/ml_assisted/region_classifier.keras")

