# This example illustrates the
# use of a basic stateful LSTM
# for stock market predictions.
#
# It also illustrates the use of
# the Keras preprocessing function to
# create time series data from pandas
# dataframe. Note that the data set
# must not be shuffled when using
# stateful LSTMs

import math
import tensorflow as tf
from tensorflow import keras
from keras import layers
import pandas as pd

# Set random seed
tf.random.set_seed(123)
# Keep a 'context' of 20 sequence steps, that is, unroll
# RNN for 20 steps
n_steps = 20
# Train for 25 epochs
n_epochs = 25

# The read the data set
data = pd.read_csv('https://evermann.ca/busi4720/djia.data.csv')

data = pd.concat([
    data,
    data.diff().add_suffix('diff'),
    data.pct_change().add_suffix('pct')],
    axis=1).iloc[1:,]
# Split data to train and validation set
# No random shuffling for time series
train = data[:math.floor(0.8*data.shape[0])]
valid = data.drop(train.index)
# Normalize data using only info from training
# set. Prevent information 'leakage'
train_mean = train.mean()
train_sd = train.std()
train = (train - train_mean)/train_sd
valid = (valid - train_mean)/train_sd

dataset_train = keras.preprocessing \
    .timeseries_dataset_from_array(
    train.drop('price.closediff', axis=1),
    train['price.closediff'],
    sequence_length=n_steps,
    sampling_rate=1, batch_size=1, shuffle=False)

dataset_valid = keras.preprocessing \
    .timeseries_dataset_from_array(
    valid.drop('price.closediff', axis=1),
    valid['price.closediff'],
    sequence_length=n_steps,
    sampling_rate=1, batch_size=1, shuffle=False)

model = keras.Sequential()
model.add(layers.InputLayer(
    batch_input_shape=(
        1, None, len(train.columns)-1)))
model.add(layers.LSTM(
    units=16,
    return_sequences=False,
    return_state=False,
    stateful=True))
model.add(layers.Dense(1))
model.summary()

model.compile(loss='mean_squared_error',
              optimizer='Adagrad')

model.fit(dataset_train,
          epochs=n_epochs,
          validation_data=dataset_valid)