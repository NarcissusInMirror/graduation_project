import numpy as np
import os
import datetime
from data_related.getdata import load
from model import multi_label_net
from resnet50_model import res50_multi_label_net
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# load data
x_train, x_valid, y_train, y_valid = load('train')

x_train = x_train.astype('float32')
x_valid  = x_valid.astype('float32')

x_train = np.reshape(x_train, (-1, 200, 200, 3))
x_valid = np.reshape(x_valid, (-1, 200, 200, 3))

# normalization
# x_train /= 255
# /= 255

# create model
model = multi_label_net()
# model = res50_multi_label_net()
# create timestamp and directory to store trained weights
current_time = datetime.datetime.now()
year = str(current_time.year)
month = str(current_time.month)
day = str(current_time.day)
hour = str(current_time.hour)
minute = str(current_time.minute)
timestamp = year + '_' + month + '_' + day + '_' + hour + '_' + minute
os.makedirs('record_files/weight_files/' + timestamp) 

# set callbacks
check = ModelCheckpoint("./record_files/weight_files/" + timestamp + "/weights.{epoch:02d}-{val_acc:.5f}.hdf5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

# start training
model.fit(x_train, y_train, batch_size=32, nb_epoch=100,callbacks=[TensorBoard(log_dir='./record_files/tensorboard_files', write_graph=True), check, reduce_lr], validation_data=(x_valid,y_valid))