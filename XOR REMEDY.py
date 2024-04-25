import numpy as np
import tensorflow as tf

# DEFINE KARDAN DATASET XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

# TOZIH SAKHTAR SHABAKE ASABI
model = tf.keras.Sequential([
    tf.keras.layers.Dense(2, input_dim=2, activation='sigmoid'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# TALFIGH KARDAN
model.compile(optimizer='adam', loss='mean_squared_error')

# TRAIN DADAN
model.fit(X, Y, epochs=1000, verbose=0)

# FINAL
print("Predictions:")
print(model.predict(X))