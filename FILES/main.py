from MODEL import AutoEn
from dataset import get_absolute_path
import tensorflow as tf
from keras.models import Model,load_model
from keras.layers import Input
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from keras.preprocessing.image import load_img,img_to_array
import keras.backend as k
from sklearn.neighbors import NearestNeighbors
import cv2
import numpy as  np
import os
from skimage.transform import resize
from skimage.io import imread
import pandas as pd
import ast 
from PIL import Image
import matplotlib.pyplot as plt

shape = [64,64,3]


NUM_EPOCHS= 1

ae = AutoEn(shape)
encoder,decoder = ae.build_model()
ae = ae.AutoEn_struct(encoder,decoder)
#print(ae.summary())

ae.compile(loss = 'mse',optimizer = 'adamax',metrics = ['mae'])

# utils function
def mycallbacks(save=None,reduce_lr=None):

    save_callback=None
    reduce_lr_callback = None
    if save:
        print('files will be saved to src/Model.h5')
        filepath = './Model_dedup.hdf5'
        save_callback = ModelCheckpoint(filepath=filepath,
                                    monitor='val_loss',
                                    save_best_only=True)
    if reduce_lr:
        print('Reduce Learning rate callback active')
        reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss',patience=2)

    return [save_callback,reduce_lr_callback]

# main  function
def process_input(filename):
    try:
        image = imread(filename)
        if len(image.shape)==4:
            image = image[0] 
        if len(image.shape)==2:
            l,h = image.shape
            image = image.reshape((l,h,1))
        if image.shape[-1] == 1:
            image = Image.open(filename).convert("RGB")
            image = np.asarray(image)
        if image.shape[-1]==4:
            image = Image.open(filename).convert('RGB')
            image= np.asarray(image)
        if image is None:
            filename_h = os.path.splitext(filename)[0]
            #filename_h = filename
            os.rename(filename,filename_h)
            print(filename)
            image = cv2.imread(filename_h)
            print(image)
        #print(filename,image.shape)
        if image is not  None:
            image = resize(image,(64,64))
            image =image/255
            image = image.astype(np.float32)
            #print(image.shape)

            return image
        else:
            return  None
    except:
        pass


#util function for notebook
def reset_session():
    k.clear_session()
    tf.reset_default_graph()
    sess = k.get_session()
    return sess

def gaussian_noise(batch_X):
    """
    adds random noise from the normal distribution to the training set
    """
    func = lambda x: x+ np.random.normal(loc=0,scale=0.1,size=x.shape)

    return func(batch_X)


#main function
def train_batches(path = '../train_images',batch_size=32):

    files = [i for i in os.listdir(path)]
    files = get_absolute_path(path,files)
    
    num_batches = int(len(files)/batch_size)
    if len(files)%batch_size!=0:
        num_batches+=1

    callbacks_my = mycallbacks(save=True,reduce_lr=True)   
    for i in range(NUM_EPOCHS):
        np.random.shuffle(files)
        for i in range(num_batches):
            batches= files[i*batch_size:i*batch_size+batch_size]
            batches_x=[]
            for i in range(len(batches)):
                x = process_input(batches[i])
                if x is not None:
                    batches_x.append(x)
            batches = np.array(batches_x)
                
            #batches = np.array(list(map(process_input,batches)))
            batches_noise = gaussian_noise(batches)
            #print(batches.shape)
            ae.fit(batches_noise,batches,callbacks = callbacks_my,validation_split=0.1)
    print('Training completed')
    return ae

# UNCOMMENT BELOW LINES TO TRAIN FROM SCRATCH
#autoencoder = train_batches()

#COMMENT THIS LINE IF ABOVE LINE IS UNCOMMENTED
autoencoder = load_model('Model_dedup.hdf5')

test_folder = '../test_images/'
folders = os.listdir(test_folder)
folders = get_absolute_path(test_folder,folders)
#print(folders)
encodings_data =pd.DataFrame()
encodings_= []
encodings__=[]
images_path = []

for folder in folders:

    #print(folder)
    #print(os.listdir(folder))
    files = get_absolute_path(folder,os.listdir(folder))
    files = [file for file in files if os.path.splitext(file)[1]=='.jpg' ]
    files = list(set(files))
    #print(files)
    for file in files:
        print(file)
        image  = process_input(file)
        batch_s = gaussian_noise(image[np.newaxis,:])
        encodings__.append(autoencoder.layers[1].predict(batch_s).reshape(-1,))
        images_path.append(file)

encodings_data['images'] = images_path
encodings_data['ENCODINGS']=encodings__
encodings_data.to_csv('unified_encodings.csv',index=False)
encodings_data.to_pickle('encodings.pkl')



























