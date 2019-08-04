from os import makedirs, listdir
from shutil import copy
from random import seed, random

def preprocess_cat_vs_dog(src_directory = './data/dogs-vs-cats/train/',dataset_home = './data/dataset_dogs_vs_cats/'):
    #Create directories
    #dataset_home = './data/dataset_dogs_vs_cats/'
    subdirs = ['test/', 'train/']
    for subdir in subdirs:
        #Create label subdirectories
        labeldirs = ['dogs/','cats/']
        for labldir in labeldirs:
            makedirs(dataset_home+subdir+labldir,exist_ok=True)
    #seed random number gen
    seed(1)
    #ratio to use for validation
    val_ratio = 0.25
    #Copy training data to subdirs
    #src_directory = './data/dogs-vs-cats/train/'
    for file in listdir(src_directory):
        src = src_directory + file
        dst_dir = 'train/'
        if random()  < val_ratio:
            dst_dir = 'test/'
        if file.startswith('cat'):
            dst = dataset_home + dst_dir + 'cats/'
            copy(src,dst)
        elif file.startswith('dog'):
            dst = dataset_home + dst_dir + 'dogs/'
            copy(src,dst)

import sys
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def define_model(targ_height,targ_width):
    #load model
    model = VGG16(include_top=False, input_shape=(targ_height,targ_width, 3))
    #mark loaded layers as nontrainable
    for layer in model.layers:
        layer.trainable=False
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128,activation='relu',kernel_initializer='he_uniform')(flat1)
    output =Dense(1,activation='sigmoid')(class1)
    #new model
    model = Model(model.inputs,outputs=output)
    #compile model
    opt = SGD(lr=0.001,momentum=0.9)
    model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])
    return model

def summarize_diagnostics(history):
    #plot loss
    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'],color='blue',label='train')
    plt.plot(history.history['val_loss'],color='orange',label='test')
    #plot accuracy
    plt.subplot(212)
    plt.title('Accuracy')
    plt.plot(history.history['acc'],color='blue',label='train')
    plt.plot(history.history['val_acc'],color='orange',label='test')
    #save
    filename = sys.argv[0].split('/')[-1]
    plt.savefig(filename+'_plot.png')
    plt.close('all')

#run the test harness for evaluating a model
def run_test_harness(epochs,targ_height,targ_width):
    #define model
    model = define_model(targ_height,targ_width)
    #create data generator
    train_datagen = ImageDataGenerator(
            rescale=1.0/255.0,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            featurewise_center=True)
    test_datagen = ImageDataGenerator(
            rescale=1.0/255.0,
            featurewise_center=True)
    
    #prepare iterators
    train_it = train_datagen.flow_from_directory(
            './data/dataset_dogs_vs_cats/train/',
            class_mode='binary',
            batch_size=64,
            target_size=(targ_height,targ_width))
    test_it = test_datagen.flow_from_directory(
            './data/dataset_dogs_vs_cats/test/',
            class_mode='binary',
            batch_size=64,
            target_size=(targ_height,targ_width))
    #fit model
    history = model.fit_generator(
            train_it,
            steps_per_epoch=len(train_it),
            validation_data = test_it,
            validation_steps=len(test_it),
            epochs=epochs,
            verbose=1)
    _,acc = model.evaluate_generator(
            test_it,
            steps=len(test_it),
            verbose=0)
    print('> %.3f' % (acc*100.0))
    #plot curves
    summarize_diagnostics(history)


targ_height = 32
targ_width = 32   
epochs = 10
run_test_harness(epochs,targ_height,targ_width)
    









"""
#Block 1
model.add(Conv2D(32,(3,3),activation='relu',kernal_initializer='he_uniform',padding='same',input_shape=(200,200,3)))
model.add(MaxPooling2D((2,2)))
#Block 2
model.add(Conv2D(64,(3,3),activation='relu',kernal_initializer='he_uniform',padding='same'))
model.add(MaxPooling2D((2,2)))
#Block 3
model.add(Conv2D(128,(3,3),activation='relu',kernal_initializer='he_uniform',padding='same'))
model.add(MaxPooling2D((2,2)))
"""