"""
Train Generator for CLARE Project
Authors: Tiago Gonçalves, Tomé Albuquerque

"""

import math
import numpy as np
import cv2
from skimage import io
import os
import pandas as pd
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

def apply_augmentation(datagen, image):
    params = datagen.get_random_transform(image.shape)
    image_tr = datagen.apply_transform(datagen.standardize(image), params)
    return image_tr


def random_saturation(image):
    saturation_factor = np.random.uniform(0.5, 2)
    image = image*saturation_factor
    ix = image > 1
    image[ix] = 1
    return image



class Generator(object):
    def __init__(self,
                 dataframe,
                 parent_directory,
                 batchsize=32,
                 use_data_aug=False,
                 rotation_range=90,
                 width_shift_range=0.2,
                 height_shift_range=0.2,
                 horizontal_flip=True,
                 zoom_range=0.3,
                 preprocessing_function = random_saturation):

        """
        Arguments
        ---------
        """
        # Images
        self.images = dataframe['image'].values

        # Labels
        self.labels = dataframe['label'].values
        self.labels = to_categorical(self.labels, num_classes=None, dtype='float32')

        # Clinical Data: Age, HPV, Time, YDT
        self.age = dataframe['age'].values.reshape((-1, 1))
        self.hpv = dataframe['hpv'].values.reshape((-1, 1))
        self.time = dataframe['time'].values.reshape((-1, 1))
        self.ydt = dataframe['ydt'].values.reshape((-1, 1))

        # Clinical Data: Complete Array
        self.clinical_data = np.hstack((self.age, self.hpv, self.time, self.ydt))

        # Train Size        
        self.size_train = dataframe.shape[0]
        
        # Parent Directory
        self.parent_directory = parent_directory

        # Batch Size
        self.batchsize = batchsize

        #Data Augmentation Parameters
        self.rotation_range=rotation_range
        self.width_shift_range=width_shift_range
        self.height_shift_range=height_shift_range
        self.horizontal_flip=horizontal_flip
        self.zoom_range=zoom_range
        self.preprocessing_function = preprocessing_function
        
        #Do or Don't Data Aug
        self.use_data_aug = use_data_aug

        # Image Data Generator
        self.data_augmentation_gen = ImageDataGenerator(
                                                        rotation_range=self.rotation_range,
                                                        width_shift_range=self.width_shift_range,
                                                        height_shift_range=self.height_shift_range,
                                                        horizontal_flip=self.horizontal_flip,
                                                        zoom_range=self.zoom_range,
                                                        preprocessing_function = self.preprocessing_function)
            


    def generate(self, batchsize=14): 
        print('Starting train...')
        print('batchsize: ',self.batchsize,)
        """Generator"""
        while True:
            cuts = [(b, min(b + self.batchsize, self.size_train)) \
            for b in range(0, self.size_train, self.batchsize)]
                
                
            for start, end in cuts:
                # Inputs -> Images
                self.inputs = np.array([io.imread(os.path.join(self.parent_directory, img_file)) for img_file in self.images[start:end].copy()])
                
                # Targets -> Labels
                self.targets = self.labels[start:end].copy()

                # Extra Regularization Data -> Clinical Data
                self.multimodal_data = self.clinical_data[start:end].copy()

                # Mini-Batch Size
                self.actual_batchsize = self.inputs.shape[0]

                #Data Augmentation
                if self.use_data_aug==True:
                    self.augmented_inputs = np.zeros_like(self.inputs)

                    for idx, img in enumerate(self.inputs):
                        self.augmented_inputs[idx] = apply_augmentation(self.data_augmentation_gen, img)
                else:
                    self.augmented_inputs = self.inputs

                
                yield ([self.augmented_inputs, self.multimodal_data], [self.targets,np.zeros((self.targets.shape[0], 28, 28, 256))])