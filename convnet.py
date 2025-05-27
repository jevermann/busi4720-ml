import plotly.subplots
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import datetime

(train_images, train_labels), \
(test_images, test_labels) \
    = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = \
train_images / 255.0, test_images / 255.0

# Show some example images
import plotly
import plotly.express as px
fig = plotly.subplots.make_subplots(rows=5, cols=5)
for i in range(25):
    fig.add_trace(px.imshow(train_images[i]).data[0], row=i // 5 + 1, col=i % 5 + 1)
fig.show(renderer='browser')

# Create a simple convolutional model:
model = models.Sequential()
model.add(layers.Conv2D(8, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (3, 3), activation='relu'))

# Add dense (fully-connected) layers for classification.
# There are 10 target classes
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(10))
# Show model summary
model.summary()

# Set up for TensorBoard:
log_dir = "./tensorboard_logs/" + \
    datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = \
    tf.keras.callbacks.TensorBoard(log_dir=log_dir,
        histogram_freq=0)

# Compile and train the model:
model.compile(optimizer='adam',
    loss=tf.keras.losses \
        .SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

history = model.fit(train_images, train_labels,
    epochs=100,
    validation_data=(test_images, test_labels),
    callbacks=[tensorboard_callback])
