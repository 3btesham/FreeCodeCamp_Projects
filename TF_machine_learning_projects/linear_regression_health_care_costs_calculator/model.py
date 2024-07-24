#%% Importing libraries and data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass
import tensorflow as tf
from tensorflow import keras
from keras import layers

dataset = pd.read_csv('https://cdn.freecodecamp.org/project-data/health-costs/insurance.csv')

#%% mapping categorical features to numeric values
dataset['sex'] = dataset['sex'].map({'female': 0, 'male': 1})
dataset['smoker'] = dataset['smoker'].map({'no': 0, 'yes': 1})
dataset['region'] = dataset['region'].map({'southwest': 0, 'southeast': 1, 'northwest': 2, 'northeast': 3})

#%% splitting the dataset into training and testing datasets
train_dataset = dataset.sample(frac=0.8)
test_dataset = dataset.drop(train_dataset.index)

#%% split the features and the labels from each other
train_labels = train_dataset.pop('expenses')
test_labels = test_dataset.pop('expenses')

#%% creating the model
model = keras.Sequential([
    keras.layers.Normalization(axis=-1),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1)
])

#%% compiing the model
model.compile(loss='mean_absolute_error',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
              metrics=['mean_absolute_error'])

#%% training the model
model.fit(train_dataset, train_labels, epochs=100, batch_size=64)

#%% RUN THIS CELL TO TEST YOUR MODEL. DO NOT MODIFY CONTENTS.
# Test model by checking how well the model generalizes using the test set.
loss, mae = model.evaluate(test_dataset, test_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} expenses".format(mae))

if mae < 3500:
  print("You passed the challenge. Great job!")
else:
  print("The Mean Abs Error must be less than 3500. Keep trying.")

# Plot predictions.
test_predictions = model.predict(test_dataset).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True values (expenses)')
plt.ylabel('Predictions (expenses)')
lims = [0, 50000]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims,lims)



