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
import pandas as pd

model = keras.models.load_model("pong_ai_model.keras")
test_data = pd.read_csv("test_data.csv")

X_test = test_data.drop("paddle_x", axis=1)
y_test = test_data["paddle_x"]


GAME_Width = 6  # from -3 to 3
PADDLE_WIDTH = 0.8

# 6 possible paddle x positions from -3 to 3
possible_paddle_x = [-2.4, -1.6, -0.8, 0, 0.8, 1.6, 2.4]
prediction = model.predict(X_test)
print(prediction)
prediction = [
    possible_paddle_x[np.argmin(np.abs(possible_paddle_x - prediction))]
    for prediction in prediction
]

print(prediction)
