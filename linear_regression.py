import pandas as pd
import tensorflow as tf

# Use the Boston housing data set
boston_data = pd.read_csv("https://evermann.ca/busi4720/boston.csv")

# Separate features and targets
boston_features = boston_data.copy()
boston_labels = boston_features.pop('medv')

# Sequential network with one hidden and one output layer
# No activation means this is a linear model
boston_model = tf.keras.Sequential([
  tf.keras.layers.Dense(64, activation=None),
  tf.keras.layers.Dense(1, activation=None)
])

# Define loss and specify optimizer
boston_model.compile(
    loss = tf.keras.losses.MeanSquaredError(),
    optimizer = tf.keras.optimizers.Adam())

# Fit model to data
boston_model.fit(boston_features, boston_labels, epochs=25)

# Show model summary
boston_model.summary()
