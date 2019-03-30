from keras.applications.vgg16 import VGG16
# from keras.applications.vgg19 import VGG19
# from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Input
from keras.optimizers import SGD, Adam, Adagrad
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import backend as K
import os
import numpy as np


def multi_label_net(pretrained_weights=None):
    
    input_tensor = Input(shape=(200, 200, 3))
    
    base_model = VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)
    
    x = base_model.output
    
    x = Flatten(name='flatten')(x)
    x = Dense(512, activation='relu', name='fc1')(x)
    x = Dense(512, activation='relu', name='fc2')(x)
    x = Dense(4, activation='sigmoid', name='prediction')(x)


    model = Model(inputs=base_model.input, outputs=x)
    
    for layer in base_model.layers:
        layer.trainable = False

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam()
    adagrad = Adagrad()

    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    if pretrained_weights != None:
        model.load_weights(pretrained_weights)
    
    return model