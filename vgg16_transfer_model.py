from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Flatten
from keras.layers import Input
from keras.optimizers import SGD
from keras import backend as K
import os
import numpy as np

def vgg16_transfer_multi_label_net(pretrained_weights=None, input_size=(200, 200, 3)):
    
    img_input = Input(shape=(200, 200, 3))
    x = BatchNormalization(axis=3, name='bn0')(img_input)
    base_model = VGG16(input_tensor=img_input, weights='imagenet', include_top=False)
    x = base_model.output
    x = Flatten(name='flatten')(x)
    x = Dense(512, activation='relu', name='fc1')(x)
    x = Dense(512, activation='relu', name='fc2')(x)
    x = Dense(4, activation='sigmoid', name='prediction')(x)

    model = Model(inputs=base_model.input, outputs=x)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    if pretrained_weights != None:
        model.load_weights(pretrained_weights)
    
    return model