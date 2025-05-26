import tensorflow as tf
from tensorflow.keras import layers, Model
from ml_training.data_loader import load_predictor_dataset

# 1. Load data
X_train, y_train, X_val, y_val = load_predictor_dataset()

# 2. Build model
inp = layers.Input(shape=(7,7,1))
x = layers.Conv2D(32, 3, activation='relu', padding='same')(inp)
x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
x = layers.Flatten()(x)
out = layers.Dense(1)(x)
model = Model(inp, out)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 3. Train
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10, batch_size=64
)

# 4. Save
model.save('models/ml_assisted/predictor.keras')
print("Saved predictor to models/ml_assisted/predictor.keras")

