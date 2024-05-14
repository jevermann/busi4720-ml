import pandas as pd
import tensorflow as tf
import plotly.express as px
import numpy as np

# Use the Boston housing data set
boston_data = pd.read_csv("https://evermann.ca/busi4720/boston.csv")

# Separate features and targets
boston_features = boston_data.copy()
boston_labels = boston_features.pop('medv')

# Normalization layer for all features
norm_layer = tf.keras.layers.Normalization()
# Adapt it to the feature data
norm_layer.adapt(boston_features.to_numpy())

# Sequential network with one hidden and one output layer
# Non-linear activation function
norm_boston_model = tf.keras.Sequential([
  norm_layer,
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(1, activation=None)
])
# Show model summary
norm_boston_model.summary()

# Define loss and specify optimizer
norm_boston_model.compile(
    loss = tf.keras.losses.MeanSquaredError(),
    optimizer = tf.keras.optimizers.Adam(),
    metrics = ['mse', 'mae'])

# Fit model to data
train_hist = norm_boston_model.fit(
    boston_features,
    boston_labels,
    batch_size=20,
    epochs=50,
    validation_split=0.33)

# Transform the training history into a suitable data frame
hist = pd.DataFrame({
    'training': train_hist.history['mse'],
    'validation': train_hist.history['val_mse']})
hist['epoch'] = np.arange(hist.shape[0])
hist = pd.melt(hist, id_vars='epoch', value_vars=['training', 'validation'])

# Plot training history
fig = px.line(hist, x='epoch', y='value', color='variable')
fig.show()
