
import os
import glob
import numpy as np
import cv2
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *
from keras import backend as K


class SiameseNetwork:

    def __init__(self, learning_rate, input_shape):
        self.learning_rate = learning_rate
        self.input_shape = input_shape
        self.FRmodel = self.build_model()


    def build_model(self):


        print('\n-----------------------------------------------------------------------------------------------------------------------------------------------')
        print('---------------------------------------------------------INITIALIZING CNN------------------------------------------------------------------------')
        print('-------------------------------------------------------------------------------------------------------------------------------------------------\n')
        print('\n “If you re going to perform inception, you need imagination. You need the simplest version of the idea-the one that will grow naturally in the subjects mind. Subtle art.” \n')
        print('------------------------------------------------------------------------------------------------------------------')
        print('------------------------------------------------------------------------------------------------------------------')
        print('Model architecture: inception_V2_resnet\n')
        print('Num inception modules = 2\n')
        print('Architecure internals:')
        print('\nFirst bock.\n')
        print('Conv2D(64, (7, 7), strides = (2, 2), ZeroPadding2D((1, 1))(X) MaxPooling2D((3, 3), strides = 2)\n')
        print('Second bock.\n')
        print('Conv2D(64, (1, 1), strides = (1, 1), BatchNormalization(axis = 1, epsilon=0.00001')
        print('inception_block_1a(X) -> inception_block_1b(X) -> inception_block_1c(X) -> Inception 2: a/b, -> inception_block_2a(X) -> inception_block_2b(X)')
        print('inception_block_3a(X) -> inception_block_3b(X), for more details about what is going on inside each moddule, check the "inception_blocks_v2" file and search using the function names ')
        print('------------------------------------------------------------------------------------------------------------------')
        print('------------------------------------------------------------------------------------------------------------------\n\n')

        def compute_triplet_loss(y_true, y_pred, alpha = self.learning_rate):
    
            anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

            pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,
                    positive)), axis=-1)
            neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, 
                    negative)), axis=-1)

            basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
            loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
        
            return loss


        K.set_image_data_format('channels_first')

        input_shape = self.input_shape
        FRmodel = faceRecoModel(input_shape)
        print('\nInput shape = ', input_shape)
        FRmodel.compile(optimizer = 'adam', loss = compute_triplet_loss, metrics = ['accuracy'])
        print('\nLoading Convolutional Layers...\n')
        load_weights_from_FaceNet(FRmodel)
        print('Done.')  
        print('Model assembled from file.')
        return FRmodel


    def generate_embedding(self, x):
         return img_to_encoding(x, self.FRmodel)




