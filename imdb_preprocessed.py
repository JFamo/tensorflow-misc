from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras

import numpy as np

import tensorflow_datasets as tfds

print(tf.__version__)

(train_data, test_data), info = tfds.load('imdb_reviews/subwords8k', split=(tfds.Split.TRAIN, tfds.Split.TEST), as_supervised=True, with_info=True)

encoder = info.features['text'].encoder

print("Vocab size: {}".format(encoder.vocab_size))

BUFFER_SIZE = 1000

train_batches = (train_data.shuffle(BUFFER_SIZE).padded_batch(32))
test_batches = (test_data.padded_batch(32))