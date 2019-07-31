### load dependencies
import time
import cv2
import os
from os.path import isfile, join
import numpy as np
import pandas as pd
import itertools

# keras backend 
from keras import backend as K

# keras nn layers
from keras.layers import Input, Dense, Activation, Reshape, Lambda, BatchNormalization, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.recurrent import GRU
from keras.layers.merge import add, concatenate
from keras.models import Model


### define model parameters
CHAR_VECTOR = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"

letters = [letter for letter in CHAR_VECTOR]

num_classes = len(letters) + 1

img_w, img_h = 128, 64

# Network parameters
batch_size = 128
val_batch_size = 16

downsample_factor = 4
max_text_len = 8


### define custom loss function
# # Loss and train functions, network architecture
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
    
    
### define model architecture
def get_model_grul4(training):
    
    #input_shape = (img_w, img_h, 1)     # (128, 64, 1)
    # adjusted by Ben 9/22
    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_w, img_h)
    else:
        input_shape = (img_w, img_h, 1)
        
    # Make Network
    inputs = Input(name='the_input', shape=input_shape, dtype='float32')  # (None, 128, 64, 1)

    # Convolution layer (VGG)
    inner = Conv2D(64, (3, 3), padding='same', name='conv1', activation='relu')(inputs)
    inner = Conv2D(64, (3, 3), padding='same', name='conv1b', activation='relu')(inner) 
    inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner)
    inner = BatchNormalization()(inner)

    inner = Conv2D(128, (3, 3), padding='same', name='conv2', activation='relu')(inner) 
    inner = Conv2D(128, (3, 3), padding='same', name='conv2b', activation='relu')(inner)
    inner = MaxPooling2D(pool_size=(2, 2), name='max2')(inner)
    inner = BatchNormalization()(inner)
    
    inner = Conv2D(256, (3, 3), padding='same', name='conv3', activation='relu')(inner) 
    inner = Conv2D(256, (3, 3), padding='same', name='conv3b', activation='relu')(inner)
    inner = MaxPooling2D(pool_size=(1, 2), name='max3')(inner)
    inner = BatchNormalization()(inner)
    
    inner = Conv2D(512, (3, 3), padding='same', name='conv4', activation='relu')(inner) 
    inner = Conv2D(512, (3, 3), padding='same', name='conv4b', activation='relu')(inner)
    inner = MaxPooling2D(pool_size=(1, 2), name='max4')(inner)
    inner = BatchNormalization()(inner)
    
    inner = Conv2D(512, (2, 2), padding='same', name='conv5', activation='relu')(inner)
    inner = BatchNormalization()(inner)
    
# #     # CNN to RNN
#     conv_to_rnn_dims = (img_w // (2 ** 2), (img_h // (2 ** 2)) * 16)
#     inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)  # (None, 32, 2048)
    inner = Reshape(target_shape=((32, 2048)), name='reshape')(inner)  # (None, 32, 2048)
    inner = Dropout(.4)(inner)
    inner = Dense(64, activation='relu', name='dense1')(inner)  # (None, 32, 64)
    inner = BatchNormalization()(inner)
    inner = Dropout(.4)(inner)
    inner = Dense(64, activation='relu', name='dense2')(inner)  # (None, 32, 64)
    
    # RNN layer
    gru_1 = GRU(512, return_sequences=True, name='gru1')(inner)
    gru_1b = GRU(512, return_sequences=True, go_backwards=True, name='gru1_b')(inner)
    gru1_merged = add([gru_1, gru_1b]) 
    gru_2 = GRU(512, return_sequences=True, name='gru2')(gru1_merged)
    gru_2b = GRU(512, return_sequences=True, go_backwards=True, name='gru2_b')(gru1_merged)
    gru2_merged = concatenate([gru_2, gru_2b])

    # transforms RNN output to character activations:
    inner = Dense(num_classes,name='dense3')(gru2_merged) #(None, 32, 63)
    y_pred = Activation('softmax', name='softmax')(inner)

    labels = Input(name='the_labels', shape=[max_text_len], dtype='float32') # (None ,8)
    input_length = Input(name='input_length', shape=[1], dtype='int64')     # (None, 1)
    label_length = Input(name='label_length', shape=[1], dtype='int64')     # (None, 1)

    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length]) #(None, 1)
    
    print("OCR model successfully configured.")

    if training:
        return Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)
    else:
        return Model(inputs=[inputs], outputs=y_pred)
        
        
