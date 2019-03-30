import numpy as np
import os
from getdata import load
from model import multi_label_net
from keras.callbacks import ModelCheckpoint, TensorBoard


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

x_test, y_test = load('test')
x_test /= 255

model = multi_label_net('/record_files/weight_files/weights.10-0.88684.hdf5')

result = model.evaluate(x_test, y_test)
metrics = model.metrics_names

for ii, mm in enumerate(metrics):
    print(mm, ':', result[ii])

