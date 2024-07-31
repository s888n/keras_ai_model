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

train_labels = []
train_samples = []

# An experimental drug was tested on individuals from ages 13 to 100
# The trial had 2100 participants. Half were under 65 years old, half were over 65 years old
# 95% of patients 65 or older experienced side effects
# 95% of patients under 65 experienced no side effects
# The data is already shuffled


# creating the sample data
# 5% of patients under 65 experienced side effects
for i in range(50):
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
    train_labels.append(1)
    # 95% of patients under 65 experienced no side effects
    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_labels.append(0)

# 95% of patients 65 or older experienced side effects
for i in range(1000):
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
    train_labels.append(0)

    # 5% of patients 65 or older experienced no side effects
    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_labels.append(1)

train_labels = np.array(train_labels)
train_samples = np.array(train_samples)

train_labels, train_samples = shuffle(train_labels, train_samples)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1, 1))

model = Sequential(
    [
        Dense(units=16, input_shape=(1,), activation="relu"),
        Dense(units=32, activation="relu"),
        Dense(units=2, activation="softmax"),
    ]
)
model.summary()

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    x=scaled_train_samples,
    y=train_labels,
    validation_split=0.1,
    batch_size=10,
    epochs=30,
    shuffle=True,
    verbose=2,
)

# # saving the model
if os.path.isfile("models/medical_trial_model.h5") is False:
    model.save("models/medical_trial_model.h5")

# # how to use the model to predict the chances of side effects in a new patient
# # - 1. create a new patient
# # - 2. scale the new patient
# # - 3. predict the new patient

# # creating a new patient
# test_samples = []
# test_labels = []

# for i in range(10):
#     random_younger = randint(13, 64)
#     test_samples.append(random_younger)
#     test_labels.append(0)

#     random_older = randint(65, 100)
#     test_samples.append(random_older)
#     test_labels.append(1)

# test_labels = np.array(test_labels)
# test_samples = np.array(test_samples)

# test_labels, test_samples = shuffle(test_labels, test_samples)

# scaled_test_samples = scaler.fit_transform(test_samples.reshape(-1, 1))

# predictions = model.predict(x=scaled_test_samples, batch_size=10, verbose=0)
# rounded_predictions = np.argmax(predictions, axis=-1)
# # print the age and the chances of side effects in %
# for i in range(20):
#     print(
#         f"Age: {test_samples[i]} -> Predicted: {rounded_predictions[i]} -> Actual: {test_labels[i]}"
#     )
