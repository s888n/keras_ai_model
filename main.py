import pandas as pd
import os.path
import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy

# load the data
data = pd.read_csv("data.csv")

# shuffle the data
data = shuffle(data)

# split the data into features and labels
X = data.drop("paddle_x", axis=1)
y = data["paddle_x"]

print(X)
print(y)

# scale the data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# create the model
model = Sequential(
    [
        Dense(units=16, input_shape=(4,), activation="relu"),
        Dense(units=32, activation="relu"),
        Dense(units=1, activation="linear"),
    ]
)

model.summary()

# compile the model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="mean_squared_error",
    metrics=["accuracy"],
)


# train the model
model.fit(
    x=X,
    y=y,
    validation_split=0.1,
    batch_size=10,
    epochs=300,
    shuffle=True,
    verbose=1,
)


# if os.path.exists("pong_ai_model.keras"):
#     os.remove("pong_ai_model.keras")
# model.save("pong_ai_model.keras")

# # testing the model
test_data = pd.read_csv("test_data.csv")
X_test = test_data.drop("paddle_x", axis=1)
y_test = test_data["paddle_x"]
X_test = scaler.transform(X_test)
predictions = model.predict(X_test)
possible_paddle_x = [-2.4, -1.6, -0.8, 0, 0.8, 1.6, 2.4]

closest_paddle_x = [
    possible_paddle_x[np.argmin(np.abs(possible_paddle_x - prediction))]
    for prediction in predictions
]
print(predictions)
print(closest_paddle_x)
