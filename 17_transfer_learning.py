# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 02:27:00 2019

@author: hp
"""

from keras.applications.vgg16 import VGG16
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

model_vgg = VGG16(weights = 'imagenet', include_top = False)
model_vgg.trainable = False
#model_vgg.summary()
global_layer = GlobalAveragePooling2D()(model_vgg.output)
prediction_layer = Dense(units = 1, activation = 'sigmoid')(global_layer)
model = Model(inputs = model_vgg.input, outputs = prediction_layer)
#model.summary()
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=["accuracy"])

train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

model.fit_generator(training_set,
                         steps_per_epoch = 8000/32,
                         epochs = 5,
                         validation_data = test_set,
                         validation_steps = 2000/32)
