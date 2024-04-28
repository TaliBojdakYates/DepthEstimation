import numpy as np
import h5py
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Input, regularizers  
import random
import os
from defs import Data_Entry, depth_loss


#helps with gpu memory allocation
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

#data roots
train_root = 'C:/Users/talib/.keras/datasets/nyudepthv2/train/'
val_root = 'C:/Users/talib/.keras/datasets/nyudepthv2/val/official'

#data generator for training
def generator(batch_size):

    #get all the names of the training data. Shuffle to help model learn
    train_names = []
    for main_folder in os.listdir(train_root):
        outer_dir = train_root + main_folder
        for inner_file in os.listdir(outer_dir):
            train_names.append(outer_dir + '/' + inner_file)
    
    random.shuffle(train_names)
    
    while True:

        image_data =  []
        depth_data = []
        
        for file in train_names:
            try:
                with h5py.File(file) as data_file:

                    entry = Data_Entry(data_file)
                    image_data.append(entry.image)
                    depth_data.append(entry.depths)
                
                #if the number of files selected equals batch size return the values for that batch
                if (len(image_data) == batch_size):
                    yield np.array(image_data), np.array(depth_data)
                    image_data =  []
                    depth_data = []

            except OSError as e:
                print(f"Error opening file: {file}")
                print(e)
                continue

#data generator for validation data
def validation_generator(batch_size):

    val_names = []
    outer_dir = val_root
    for inner_file in os.listdir(outer_dir):
        val_names.append(outer_dir + '/' + inner_file)
    
  
    while True:

        image_data =  []
        depth_data = []
    
        for file in val_names:
            try:
                with h5py.File(file) as data_file:

                    entry = Data_Entry(data_file)
                    image_data.append(entry.image)
                    depth_data.append(entry.depths)
                
                if (len(image_data) == batch_size):

                    yield np.array(image_data), np.array(depth_data)
                    image_data =  []
                    depth_data = []

            except OSError as e:
                print(f"Error opening file: {file}")
                print(e)
                continue


def depth_model():

    input_shape = (480,640,3)
    inputs = Input(shape=input_shape) 

    encoder = tf.keras.applications.ResNet101V2(
    include_top=False, weights='imagenet', input_tensor=inputs
    )


 
    x = encoder.output
    x = layers.Conv2DTranspose(1024, 3, strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(512, 3, strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(256, 3, strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(128, 3, strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(64, 3, strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(32, 3, strides=(1, 1), padding='same', activation='relu')(x)
    x = layers.Conv2D(1, 1, activation=None)(x)  # Output layer

    # Define the model
    outputs = x
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    print(model.summary())


    model.compile(optimizer='adam', loss=depth_loss)
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=3)
    history = model.fit(generator(batch_size), steps_per_epoch=steps_per_epoch,  validation_data=validation_generator(batch_size), validation_steps=val_steps, epochs=8, callbacks =[early_stopping] ,verbose=1)

    model.save("ResNet101_Regularization_Model.keras")
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    

    return model


#get all the training data names
train_names = []
for main_folder in os.listdir(train_root):
    outer_dir = train_root + main_folder
    for inner_file in os.listdir(outer_dir):
        train_names.append(outer_dir + '/' + inner_file)
        
#get all the testing data names
val_names = []
for main_folder in os.listdir(val_root):
    val_names.append(val_root + '/' + main_folder)


batch_size = 4

steps_per_epoch = int((len(train_names) / batch_size) + 0.5)
val_steps = int((len(val_names) / batch_size) + 0.5)

depth_model()





