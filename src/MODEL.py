from architecture import Architecture
from keras.layers import Reshape, Dense,Flatten,Input
from keras.models import Model



class AutoEn(Architecture):

    def __init__(self,shape):

        self.shape = shape
        self.mtype = 'vision'
        self.strides = 2
        self.kernel_size=3
        self.filters = [32,64,128,256]
        self.conv_base_name = 'conv'
        self.pool_base_name = 'pool'
        self.upsample_name = 'upsample'
        self.code_size = 128
        self.h = None
        self.w = None
        self.c = None

        super().__init__(self.mtype)

    def build_model(self):

        """
        defines encoder
        and 
        decoder 
        """

        #encoder architecture

        inp_e = Input(shape=self.shape)
        x = self.convolution_layer(self.filters[0],
                                   self.kernel_size,
                                   stride =self.strides,
                                   name=self.conv_base_name+str(0))(inp_e)
       
        #loop through remaining filters 
        for i,filter in enumerate(self.filters[1:]):
            x = self.convolution_layer(filter,
                                   self.kernel_size,
                                   stride =self.strides,
                                    name=self.conv_base_name+str(i+1))(x)
        
        # get the dimensions of last convolution layer
        _,self.h,self.w,self.c = x.shape.as_list()

        x = Flatten()(x)

        #this is the embedding tensor 
        x = Dense(self.code_size)(x)

        encoder =  Model(inputs = inp_e,outputs = x)

        #decoder architecture

        inp_d = Input(shape = [self.code_size,])
        hidden_dim1 = self.h*self.w*self.c
        x = Dense(hidden_dim1)(inp_d)

        #reshape to make 4 dimensional
        x = Reshape((self.h,self.w,self.c))(x)

        filters = self.filters[-2::-1]
        out_channel = 3
        filters.append(out_channel)

        for i,filter in enumerate(filters):
            
            x = self.upsample(filter,
                              self.kernel_size,
                              self.strides,
                              name = self.upsample_name+str(i))(x)

        #decoder model
        decoder = Model(inputs = inp_d,outputs = x)

        return encoder,decoder
    

    def AutoEn_struct(self,encoder,decoder):

        """
        exposed api used
        """

        
        #print(encoder.summary(),decoder.summary())

        Inp_ae = Input(shape=self.shape)
        encoding = encoder(Inp_ae)
        decoding = decoder(encoding)
        
        ae = Model(inputs = Inp_ae,outputs=decoding)

        return ae

    

    




    
