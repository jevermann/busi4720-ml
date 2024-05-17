# This example illustrates the use of LSTM
# networks for business process predictions.
# The task is a classification task to predict
# the next process activity from a sequence of
# some prior activities.

# The example log file is taken from here:
# https://doi.org/10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f
# where it has been published under the 4TU general terms of use
# https://data.4tu.nl/articles/_/12721292/1

import numpy
from tensorflow import keras
from keras import layers
import pandas as pd
import pm4py

# Length of sequences to predict from
prefix_len= 5

# Read the log
log = pm4py.read_xes('BPI_Challenge_2012.xes.gz')

log['time:timestamp'] = pd.to_datetime(log['time:timestamp'], utc=True)
log['case:REG_DATE'] = pd.to_datetime(log['case:REG_DATE'], utc=True)
log['case:AMOUNT_REQ'] = pd.to_numeric(log['case:AMOUNT_REQ'])
log['org:resource'] = log['org:resource'].astype(str)

# Retain only activity completion events
log = log[log['lifecycle:transition'] == 'COMPLETE']

# Find the case start time as time of the
# first event in case
log = log.merge(
    log.groupby('case:concept:name', as_index=False) \
        ['time:timestamp'].min() \
        .rename(columns={'time:timestamp': 'case:start'}),
    how='left')

# Find the number of events for each case
log = log.merge(
    log.groupby('case:concept:name', as_index=False) \
        ['time:timestamp'].count(). \
        rename(columns={'time:timestamp': 'num_events'}),
    how='left')

# Filter log for minimum 6 events (5 input, 1 target)
log = log[log['num_events'] > prefix_len]

# Sort log by case start, then by event time
log.sort_values(['case:start', 'time:timestamp'], inplace=True)

# This is the feature we predict (from)
f_name = 'concept:name'

# Vocabulary
vocab = list(log[f_name].unique()) + ['EOC']
v_size = len(vocab)

# Dictionaries to convert to/from integers
f2int = dict([(s, vocab.index(s)) for s in vocab])
int2f = dict([(v, k) for (k, v) in f2int.items()])

# Make sequences of feature names for each case
features = log.groupby(['case:concept:name'])[f_name] \
    .apply(list).reset_index(name='features')

sequences=[l+['EOC'] for l in list(features['features'])]
sequences=[[f2int[i] for i in seq] for seq in sequences]

data = pd.DataFrame([(seq[i:i+prefix_len], \
    seq[i+prefix_len]) for seq in sequences \
    for i in range(len(seq)-prefix_len)])

# Split the lists into dataframe columns
data = data.assign(**data[0] \
    .apply(pd.Series).add_prefix('index_'))

# Divide into train and test set
train = data.sample(frac=0.8)
valid = data.drop(train.index)

# Separate X and Y
train_x = train.iloc[:,2:]
train_y = train.iloc[:,1]
valid_x = valid.iloc[:,2:]
valid_y = valid.iloc[:,1]

model = keras.Sequential()
model.add(layers.InputLayer(input_shape=(5,)))
model.add(layers.Embedding(input_dim=v_size,
                           output_dim=16))
model.add(layers.LSTM(units=32,
                      return_sequences=False,
                      return_state=False,
                      stateful=False))
model.add(layers.Dense(v_size, activation='softmax'))
model.summary()

# Compile with loss and optimizer
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='Adagrad',
              metrics=['sparse_categorical_accuracy'])

# Train the model and validate on
model.fit(train_x, train_y,
          validation_data=(valid_x, valid_y),
          epochs=25, shuffle=True)

# ###
# Predicting from the trained model
# ###

# Take an example from the training set as input
input = train_x.iloc[2:3,:].copy()
print(input)

# Take the probabilities of the first entry of the return batch
probs = model.predict(input)[0]

# Either deterministically choose the most probable class
# pred = probs.argmax()
# Better is to randomly sample from the probabilities
pred = numpy.random.choice(a=range(v_size), p=probs)

# Print the result
print(int2f[pred])

# And keep doing this until end-of-case is reached
while int2f[pred] != 'EOC':
    for i in range(4):
        input.iat[0,i] = input.iat[0, i+1]
    input.iat[0,4] = pred
    print(input)
    probs = model.predict(input)[0]
    # pred = probs.argmax()
    pred = numpy.random.choice(a=range(v_size),p=probs)
    print(int2f[pred])
