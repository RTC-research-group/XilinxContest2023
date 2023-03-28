import tensorflow as tf
import math
import numpy as np
import pandas as pd

class BBC_CNN_sequence(tf.keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
        self.batch_size]

        return np.array(batch_x), np.array(batch_y)

model = tf.keras.models.load_model('models/CNN/weights.100-23.14.hdf5',compile=False)
model.compile()
from tensorflow_model_optimization.quantization.keras import vitis_quantize

train_data = np.load('encoded_data.npy')
train_labels = np.load('labels.npy')


n = np.max(train_labels) +1
one_hot_labels = np.eye(n)[train_labels] 

calib_dataset = BBC_CNN_sequence(x_set=train_data.tolist(),y_set= one_hot_labels,batch_size=16)

quantizer = vitis_quantize.VitisQuantizer(model)
quantized_model = quantizer.quantize_model(calib_dataset=calib_dataset, 
                                           calib_steps=100, 
                                           calib_batch_size=10,
                                           ) 

quantized_model.save('quantized_model.hdf5')