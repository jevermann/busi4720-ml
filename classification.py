import pandas as pd
import tensorflow as tf
import plotly.express as px
import numpy as np
import datetime

# This uses Keras 2
# Developed on Tensorflow 2.14.0
import os;os.environ["TF_USE_LEGACY_KERAS"]="1"

# Read data and separate features from target labels
wage_data = pd.read_csv("https://evermann.ca/wage.csv")
wage_features = wage_data.copy()
wage_labels = wage_features.pop('wagequart') - 1

# Keep track of inputs and preprocessed inputs
inputs = {}
preproc_inputs = []

# Loop over all categorical features
for cat_feature in ['maritl', 'race', 'education', 'jobclass', 'health', 'health_ins']:
    # An Input variable is a placeholder that
    # accepts data input when training or predicting
    input = tf.keras.Input(shape=(1,), name=cat_feature, dtype=tf.string)
    # This StringLookup layer accepts a string and
    # outputs its category as a one-hot vector (or,
    # alternatively as an integer)
    lookup = tf.keras.layers.StringLookup(
        name=cat_feature + "_lookup", output_mode="one_hot")
    # Adapt it to the different strings in the data
    lookup.adapt(wage_features[cat_feature])
    # And tie the input to this layer
    onehot = lookup(input)
    # Add the input feature to the list of inputs
    inputs[cat_feature] = input
    # Append the preprocessed feature to the list of preprocessed inputs
    preproc_inputs.append(onehot)

# Normalize the age feature
age_input = tf.keras.Input(shape=(1,), name="age", dtype="float32")
norm_layer = tf.keras.layers.Normalization(name="age_norm")
norm_layer.adapt(wage_features["age"])
age_norm = norm_layer(age_input)
# Append to collection of inputs and preprocessed inputs
inputs["age"] = age_input
preproc_inputs.append(age_norm)

# Treat the year as categorical with one-hot encoding
# Define the input placeholder
year_input = tf.keras.Input(shape=(1,), name="year", dtype="int32")
# Define and adapt an IntegerLookup layer for the one-hot encoding
lookup = tf.keras.layers.IntegerLookup(name="year_lookup", output_mode="one_hot")
lookup.adapt(wage_features["year"])
year_onehot = lookup(year_input)
# Add the input and preprocessed input to the collections
inputs["year"] = year_input
preproc_inputs.append(year_onehot)

# Concatenate all inputs
preprocessed_inputs = tf.keras.layers.Concatenate(name="concat", axis=1)(preproc_inputs)
# Define the preprocessing model
preproc_model = tf.keras.Model(inputs, preprocessed_inputs, name="preproc")
preproc_model.summary()

# Build the classification model, with non-linear activation and dropout
class_model = tf.keras.Sequential(name="classification")
class_model.add(tf.keras.layers.Dense(64, activation="relu"))
class_model.add(tf.keras.layers.Dropout(0.25))
class_model.add(tf.keras.layers.Dense(32, activation="relu"))
class_model.add(tf.keras.layers.Dropout(0.25))
class_model.add(tf.keras.layers.Dense(4, activation="softmax"))
# Alternatively:
# class_model.add(tf.keras.layers.Dense(4, activation=None))
# class_model.add(tf.keras.layers.Softmax())

# The inputs to the classification model are the outputs of the preprocessing model for the raw input
class_results = class_model(preproc_model(inputs))
class_model.summary()

# The total model to predict wages has raw features as input and classification results as output
wage_model = tf.keras.Model(inputs, class_results, name="wage_model")
wage_model.summary()

# Compile the model with loss, optimizer and metrics
wage_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1 = 0.9,
        beta_2 = 0.999,
        epsilon = 1e-07),
    metrics=[
        tf.keras.metrics.SparseCategoricalAccuracy(),
        tf.keras.metrics.KLDivergence()])
# Note: Specifying from_logits=True for the loss function can save
# the softmax activation or the softmax layer at the bottom of the
# sequential classification model.

wage_feature_dict = \
    {name: np.array(value) for \
        name, value in wage_features.items()}

log_dir = "./tensorboard_logs/" + \
    datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = \
    tf.keras.callbacks.TensorBoard(log_dir=log_dir,
        histogram_freq=0)

wage_hist = wage_model.fit(
    x = wage_feature_dict,
    y = wage_labels,
    validation_split=0.333,
    batch_size=20,
    epochs = 25,
    callbacks=[tensorboard_callback])

# Issue the following command from your shell:
# tensorboard --logdir tensorboard_logs
# Then browse to http://locahost:6006

hist = pd.DataFrame({
    'training': wage_hist.history['sparse_categorical_accuracy'],
    'validation': wage_hist.history['val_sparse_categorical_accuracy']})
hist['epoch'] = np.arange(hist.shape[0])
hist = pd.melt(hist, id_vars='epoch', value_vars=['training', 'validation'])

fig = px.line(hist, x='epoch', y='value', color='variable')
fig.show()
