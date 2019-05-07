
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
        self.FRmodel = build_model()
        self.input_shape = input_shape


    def compute_triplet_loss(self, y_true, y_pred, alpha = self.learning_rate):

        anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,
                positive)), axis=-1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, 
                negative)), axis=-1)

        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
        loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    
        return loss


    def build_model(self):
        K.set_image_data_format('channels_first')
        input_shape = self.input_shape
        FRmodel = faceRecoModel(input_shape)
        print('\nInput shape = ', input_shape)
        FRmodel.compile(optimizer = 'adam', loss = compute_triplet_loss, metrics = ['accuracy'])
        print('Loading weights..\n\n')
        load_weights_from_FaceNet(FRmodel)
        print('Done.')  
        return FRmodel


    def generate_embeddings():

        print('--------------------------------------------------------------------------')
        print('----------------------GENERATING EMBEDDINGS-------------------------------')
        print('--------------------------------------------------------------------------')
        database = {}

        for file in glob.glob("cropped_images/*"):
        
            identity = os.path.splitext(os.path.basename(file))[0]
            database[identity] = img_path_to_encoding(file, self.FRmodel)
        
        print('\nDone.')
        print('Saving...')
        np.savetxt('embeddings.csv', embeddings, delimiter = ',')



if __name__ == '__main__':

    facenet = SiameseNetwork(0.25, (3, 96, 96))
    facenet.generate_embeddings()