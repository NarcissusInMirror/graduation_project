import numpy as np
import os
from data_related.getdata import load
from model import multi_label_net
from keras.callbacks import ModelCheckpoint, TensorBoard


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# load data
x_train, x_test, y_train, y_test = load('train')

x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')

x_train = np.reshape(x_train, (-1, 200, 200, 3))
x_test = np.reshape(x_test, (-1, 200, 200, 3))

# normalization
x_train /= 255
x_test /= 255

# create model
model = multi_label_net()

# set callbacks
check = ModelCheckpoint("./record_files/weight_files/weights.{epoch:02d}-{val_acc:.5f}.hdf5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')

# start training
model.fit(x_train, y_train, batch_size=32, nb_epoch=50,callbacks=[TensorBoard(log_dir='./record_files/tensorboard_files', write_graph=True), check],validation_data=(x_test,y_test))