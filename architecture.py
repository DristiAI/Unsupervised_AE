import keras
from keras.layers import Conv2D,Conv2DTranspose,Reshape
from keras.layers import MaxPooling2D,AveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import Dense,Flatten,Input,Flatten

class Architecture(object):

    def __init__(self,mtype):
        self.mtype = mtype
        if self.mtype == 'vision':
            print("Vision mode activated...")
        else:
            print('text mode activated...')
         

    def convolution_layer(self,filters,kernel_size,name=None,stride=None):

        """

        Vision encoder util

        """

        if stride:
            return Conv2D(filters,
                          kernel_size,
                          name=name,
                          strides=stride,
                          padding='same',
                          activation='elu',
                          )
        return Conv2D(filters,
                      kernel_size,
                      name=name,
                      strides=2,
                      padding='same',
                      activation='elu'
                      )

    def pooling_layer(self,name,type='max'):

        """
        Vision encoder util
        options to choose
        'type' of pooling layer
        
        """

        if type=='max':
            return MaxPooling2D(pool_size=(2,2),
                                padding='same',
                                name=name)

        elif type == 'average':
            return AveragePooling2D(pool_size=(2,2),
                                    padding='same',
                                    name=name)
    
    def upsample(self,filters,kernel_size,stride=None,name=None):

        """

        Vision decoder util

        """
        if stride:
            return keras.layers.Conv2DTranspose(filters,
                                                kernel_size,
                                                strides=stride,
                                                activation='elu',
                                                padding='same',
                                                name=name
                                                )
        return keras.layers.Conv2DTranspose(filters,
                                     kernel_size,
                                     strides=2,
                                     activation='elu',
                                     padding='same',
                                     name=name)

 
    def encoding_out(self,last_encoded_layer,code_size):

        """
        Visual decoder util

        Returns 
        Image encodings: used for image representations
        code_size : dimensionality of encodings
        dims : height, width , channel size of last conv layer

        """
        _,h,w,c= last_encoded_layer.shape.as_list()
        x=last_encoded_layer
        x= Flatten()(x)
        x = Dense(code_size)(x)
        return x,code_size,(h,w,c)

    def decoder_input(self,code_size,dims):

        """
        Visual decoder util

        """

        h,w,c = dims
        x= Input(shape=[code_size,])
        x= Dense(h*w*c)(x)
        x= Reshape((h,w,c))(x)
        return 
    

    
    



    

        

