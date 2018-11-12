from architecture import Architecture
from keras.layers import Reshape, Dense,Flatten,Input,BatchNormalization,LSTM
from keras.models import Model



class AutoEn(Architecture):

    def __init__(self,shape,mtype):

        self.shape = shape
        self.mtype = mtype
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
        inp_e = Input(shape=self.shape)
        H,W,C = inp_e.shape.as_list()
        outD = Reshape((-1,1))(inp_e) 
        #h,w = outD.shape.as_list()
        outD1 = LSTM(50,return_sequences=True)(outD)
        outD1 = BatchNormalization()(outD1)
        outD1 = LSTM(50)(outD1)
        outD1 = BatchNormalization()(outD1)
        outD1 = Dense(128,activation='relu')(outD1)
        encoder =  Model(inputs = inp_e,outputs = outD1)

        outD = Dense(128,activation='relu')(outD1)
        outD1 = BatchNormalization(outD)
        outD1 = Reshape((-1,1))(outD1)
        outD1 = LSTM(50,return_sequences=True)(outD1)
        outD1 = BatchNormalization()(outD1)
        outD1 = Dense(H*W*C,activation='relu')
        outD1 = BatchNormalization()(outD1)
        outD2 = Reshape((H,W,C))(outD1)
        decoder = Model(inputs = outD,outputs = outD2)

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