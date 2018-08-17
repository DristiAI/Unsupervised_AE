from MODEL import AutoEn
from dataset import get_absolute_path
import tensorflow as tf
from keras.models import Model
from keras.layers import Input
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from keras.preprocessing.image import load_img,img_to_array
import keras.backend as k
import cv2
import numpy as  np
import os
from skimage.transform import resize

shape = [64,64,3]


NUM_EPOCHS= 12

ae = AutoEn(shape)
ae = ae.AutoEn_struct()
#print(ae.summary())

ae.compile(loss = 'mse',optimizer = 'adamax',metrics = ['mae'])

# utils function
def callbacks(save=None,reduce_lr=None):

    save_callback=None
    reduce_lr_callback = None
    if save:
        filepath = 'Model-{epoch:02d}-{val_loss:.2f}.hdf5'
        save_callback = ModelCheckpoint(filepath=filepath,
                                    monitor='val_loss',
                                    save_best_only=True)
    if reduce_lr:
        reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss',patience=10)

    return [save_callback,reduce_lr_callback]

# main  function
def process_input(filename):

    image = cv2.imread(filename)
    if image is None:
        filename = os.path.splitext(filename)[0]
        filename = filename+'.png'
        image = cv2.imread(filename)
    #print(filename,image.shape)
    image = resize(image,(64,64))
    image =image/255
    image = image.astype(np.float32)
    #print(image.shape)

    return image


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
    func = lambda x: x+np.random.normal(loc=0,scale=0.1,size=x.shape)

    return func(batch_X)


#main function
def train_batches(path = '../train_images',batch_size=32):

    files = [i for i in os.listdir(path) if os.path.splitext(i)[1]=='.jpg']
    files = get_absolute_path(path,files)
    
    num_batches = int(len(files)/batch_size)
    if len(files)%batch_size!=0:
        num_batches+=1
        
    for i in range(NUM_EPOCHS):
        np.random.shuffle(files)
        for i in range(num_batches):
            batches= files[i*batch_size:i*batch_size+batch_size]
            batches = np.array(list(map(process_input,batches)))
            batches_noise = gaussian_noise(batches)
            #print(batches.shape)
            ae.fit(batches_noise,batches)

    

train_batches()





