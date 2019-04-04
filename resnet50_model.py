from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import Input
from keras.optimizers import SGD
from keras import backend as K
import os
import numpy as np

def res50_multi_label_net(pretrained_weights=None, input_size=(200, 200, 3)):
    
    input_tensor = Input(shape=(200, 200, 3))
    base_model = ResNet50(input_tensor=input_tensor, weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(4, activation='sigmoid', name='fc4')(x)

    model = Model(inputs=base_model.input, outputs=x)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    if pretrained_weights != None:
        model.load_weights(pretrained_weights)
    
    return model