# ============================================
# CELL 1: Import Required Libraries
# ============================================
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import os

try:
    np.set_printoptions(precision=3, suppress=True)
except:
    pass

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(f"TensorFlow version: {tf.__version__}")

# ============================================
# CELL 2: Load and Explore the Dataset
# ============================================
if not os.path.exists('insurance.csv'):
    !wget https://cdn.freecodecamp.org/project-data/health-costs/insurance.csv

dataset = pd.read_csv('insurance.csv')

# ============================================
# CELL 3: Data Preprocessing
# ============================================
dataset_processed = dataset.copy()
dataset_processed['sex'] = dataset_processed['sex'].map({'male': 1, 'female': 0})
dataset_processed['smoker'] = dataset_processed['smoker'].map({'yes': 1, 'no': 0})
dataset_processed = pd.get_dummies(dataset_processed, columns=['region'], prefix='region')

# ============================================
# CELL 4: Split Data into Train and Test Sets
# ============================================
train_dataset = dataset_processed.sample(frac=0.8, random_state=0)
test_dataset = dataset_processed.drop(train_dataset.index)

train_labels = train_dataset.pop('expenses')
test_labels = test_dataset.pop('expenses')

# ============================================
# CELL 5: Build the Neural Network Model with Normalization
# ============================================
# Normalization is key to getting MAE below 3500
normalizer = layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_dataset))

model = keras.Sequential([
    normalizer,
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mae',
    metrics=['mae', 'mse']
)

# ============================================
# CELL 6: Train the Model
# ============================================
model.fit(train_dataset, train_labels, epochs=100, validation_split=0.2, verbose=0)

# ============================================
# CELL 7: Evaluate the Model
# ============================================
loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=0)
print(f"\nMean Absolute Error: ${mae:.2f}")

if mae < 3500:
    print("\u2705 SUCCESS! Model predicts healthcare costs within $3500.")
else:
    print(f"\u274c Model MAE is {mae:.2f}, needs to be under $3500.")
