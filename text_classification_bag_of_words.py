import collections
import pathlib
import datetime

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import utils
from tensorflow.keras.layers import TextVectorization

# Get the data
data_url = 'http://download.tensorflow.org/data/stack_overflow_16k.tar.gz'
dataset_dir = utils.get_file(origin=data_url, untar=True, cache_subdir='stack_overflow')

# Remember where we put it
dataset_dir = pathlib.Path(dataset_dir).parent
train_dir = dataset_dir/'train'
test_dir = dataset_dir/'test'

# Print a sample
# sample_file = train_dir/'python/1755.txt'
# with open(sample_file) as f:
#   print(f.read())

raw_train_ds = utils.text_dataset_from_directory(
    train_dir, batch_size=32,
    validation_split=0.2,
    subset='training', seed=42)

raw_val_ds = utils.text_dataset_from_directory(
    train_dir, batch_size=32,
    validation_split=0.2,
    subset='validation', seed=42)

# Read the test set portion of the dataset:
raw_test_ds = utils.text_dataset_from_directory(
    test_dir, batch_size=32)

# Use the Keras TextVectorization pre-processing layer:
multi_hot_vectorize_layer = TextVectorization(
    max_tokens=10000,
    standardize='lower_and_strip_punctuation',
    split='whitespace',
    output_mode='multi_hot')

# Get only the text from the training set
train_text = raw_train_ds.map(lambda text, labels: text)
# Adapt the layer
multi_hot_vectorize_layer.adapt(train_text)

# Print a sample:
# Retrieve a batch from the dataset
# text_batch, label_batch = next(iter(raw_train_ds))
# Applying the text vectorization layer
# to the first example and print its output
# print(text_batch[0])
# print(list(multi_hot_vectorize_layer(text_batch[0]).numpy()))

# Define a simple model, set loss function,
# optimizer and a metric to collect:
multi_hot_model = tf.keras.Sequential([
    multi_hot_vectorize_layer,
    layers.Dense(4)]
)

multi_hot_model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy'])

multi_hot_model.fit(raw_train_ds,
                    validation_data=raw_val_ds,
                    epochs=25)
