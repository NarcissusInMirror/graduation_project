from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input
from keras.optimizers import SGD, Adam, Adagrad
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
from keras import backend as K
import os
import numpy as np


def multi_label_net(pretrained_weights=None, input_size = (200, 200, 3)):

    img_input = Input(input_size)    
    # Block 1
    x = BatchNormalization(axis=3, name='bn0')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_regularizer=regularizers.l2(0.01))(x)
    x = BatchNormalization(axis=3, name='bn1')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_regularizer=regularizers.l2(0.01))(x)
    x = BatchNormalization(axis=3, name='bn2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_regularizer=regularizers.l2(0.01))(x)
    x = BatchNormalization(axis=3, name='bn3')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_regularizer=regularizers.l2(0.01))(x)
    x = BatchNormalization(axis=3, name='bn4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_regularizer=regularizers.l2(0.01))(x)
    x = BatchNormalization(axis=3, name='bn5')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_regularizer=regularizers.l2(0.01))(x)
    x = BatchNormalization(axis=3, name='bn6')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_regularizer=regularizers.l2(0.01))(x)
    x = BatchNormalization(axis=3, name='bn7')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_regularizer=regularizers.l2(0.01))(x)
    x = BatchNormalization(axis=3, name='bn8')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_regularizer=regularizers.l2(0.01))(x)
    x = BatchNormalization(axis=3, name='bn9')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', kernel_regularizer=regularizers.l2(0.01))(x)
    x = BatchNormalization(axis=3, name='bn10')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', kernel_regularizer=regularizers.l2(0.01))(x)
    x = BatchNormalization(axis=3, name='bn11')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', kernel_regularizer=regularizers.l2(0.01))(x)
    x = BatchNormalization(axis=3, name='bn12')(x)
    x = Conv2D(512, (3, 3), activation='relu',  padding='same', name='block5_conv3', kernel_regularizer=regularizers.l2(0.01))(x)
    x = BatchNormalization(axis=3, name='bn13')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    # Top layers
    x = Flatten(name='flatten')(x)
    x = Dense(512, activation='relu', name='fc1', kernel_regularizer=regularizers.l2(0.01))(x)
    x = Dense(512, activation='relu', name='fc2', kernel_regularizer=regularizers.l2(0.01))(x)
    x = Dense(4, activation='sigmoid', name='prediction', kernel_regularizer=regularizers.l2(0.01))(x)

    # Create model.
    model = Model(img_input, x, name='label_net')
    
#     for layer in base_model.layers:
#         layer.trainable = False

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr=0.01)
    adagrad = Adagrad()
    
    def mycrossentropy(y_true, y_pred, e=0.1):
        loss1 = K.binary_crossentropy(y_true, y_pred)
        loss2 = K.binary_crossentropy(K.ones_like(y_pred)/nb_classes, y_pred)
        return (1-e)*loss1 + e*loss2
    
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    if pretrained_weights != None:
        model.load_weights(pretrained_weights)
    
    return model