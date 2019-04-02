import numpy as np
import os
from data_related.getdata import load
from model import multi_label_net
from keras.callbacks import ModelCheckpoint, TensorBoard


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# load the data and reshape
x_test, y_test = load('test')
x_test = np.reshape(x_test, (-1, 200, 200, 3))

# list all the weight you want to test in the list, the code will 
# test each of them in order
# model_list = ['weights.01-0.71406.hdf5', 'weights.02-0.78359.hdf5', 'weights.03-0.80391.hdf5', 'weights.04-0.82188.hdf5', 'weights.05-0.83672.hdf5', 'weights.09-0.84453.hdf5', 'weights.10-0.84609.hdf5', 'weights.12-0.85859.hdf5', 'weights.16-0.86875.hdf5', 'weights.18-0.87031.hdf5', 'weights.26-0.87344.hdf5', 'weights.32-0.87422.hdf5', 'weights.33-0.87891.hdf5']

WEIGHT_PATH = './record_files/weight_files/2019_4_2_10_57/'
model_list = os.listdir(WEIGHT_PATH)

for i in model_list:
    if i[-4:] == 'hdf5':
        model = multi_label_net(WEIGHT_PATH + i)
        result = model.evaluate(x_test, y_test)
        print('using weight', i)
        metrics = model.metrics_names
        for ii, mm in enumerate(metrics):
            print(mm, ':', result[ii])
    else:
        pass
    
