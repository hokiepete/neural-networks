from os import makedirs, listdir
from shutil import copy
from random import seed, random

def preprocess_cat_vs_dog_final(src_directory = './data/dogs-vs-cats/train/',dataset_home = './data/dataset_dogs_vs_cats-final/'):
    #Create directories
    #dataset_home = './data/dataset_dogs_vs_cats/'
    labeldirs = ['dogs/','cats/']
    for labldir in labeldirs:
        makedirs(dataset_home+labldir,exist_ok=True)
    #Copy training data to subdirs
    #src_directory = './data/dogs-vs-cats/train/'
    for file in listdir(src_directory):
        src = src_directory + file
        if file.startswith('cat'):
            dst = dataset_home + 'cats/'
            copy(src,dst)
        elif file.startswith('dog'):
            dst = dataset_home + 'dogs/'
            copy(src,dst)

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
    drop1 = Dropout(0.5)(class1)
    output =Dense(1,activation='sigmoid')(drop1)
    #new model
    model = Model(model.inputs,outputs=output)
    #compile model
    opt = SGD(lr=0.001,momentum=0.9)
    model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])
    return model

#run the test harness for evaluating a model
def run_test_harness(epochs,targ_height,targ_width):
    #define model
    model = define_model(targ_height,targ_width)
    #create data generator
    datagen = ImageDataGenerator(
            rescale=1.0/255.0,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            featurewise_center=True)
    #prepare iterators
    train_it = datagen.flow_from_directory(
            './data/dataset_dogs_vs_cats-final',
            class_mode='binary',
            batch_size=64,
            target_size=(targ_height,targ_width))
    #fit model
    model.fit_generator(
            train_it,
            steps_per_epoch=len(train_it),
            epochs=epochs,
            verbose=1)
    #save
    model.save('./models/final_cnn_dvc_trans.h5')



targ_height = 32
targ_width = 32   
epochs = 10
#preprocess_cat_vs_dog_final()
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