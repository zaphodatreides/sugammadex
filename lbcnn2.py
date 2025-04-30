from keras.models import Sequential,Model
from keras.layers import Dropout,Input
from keras.layers import TimeDistributed,Flatten
from keras.layers.core import Dense, Activation
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint,Callback, EarlyStopping
from keras import optimizers
from keras.layers import GlobalAveragePooling2D,TimeDistributed,ZeroPadding2D,Bidirectional,LayerNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D,AveragePooling2D
from keras.models import load_model 

def returnName():
 return "PicCNN2d"

def buildModel(largest,features,classes):
 input1=Input(shape=(largest,features*6,1))
 #input2=Input(shape=(largest,1))
 conv1=Conv2D(filters=20,kernel_size=(50,4), activation='relu',padding='valid')(input1) #(16/8/4)
 pool1=MaxPooling2D(pool_size=6,padding='same')(conv1)
 pool1=LayerNormalization(axis=-1)(pool1)
 conv2=Conv2D(filters=36,kernel_size=(20,3),padding='valid')(pool1)
 #pool1=MaxPooling2D(pool_size=3,padding='same')(conv1)
 pool2=MaxPooling2D(pool_size=6,padding='same')(conv2)
 pool2=LayerNormalization(axis=-1)(pool2)
 conv3=Conv2D(filters=72,kernel_size=(10,1),padding='valid')(pool2)
 #pool1=LayerNormalization(axis=-1)(pool1)
 #conv3=Conv2D(filters=24, kernel_size=(1,1), activation='relu',padding='valid')(pool2)
 #conv3=Conv2D(filters=64, kernel_size=(9,9), activation='relu',padding='same') (pool2) #kernel size was 8/4/2
 conv3=MaxPooling2D(pool_size=6,padding='same')(conv3)
 conv3=LayerNormalization(axis=-1)(conv3)


 drop=Flatten()(conv3)
 #drop=GlobalAveragePooling1D()(drop) 
 #path1Shape=drop.get_shape()
 
 #dense1=LSTM(16,return_sequences=True)(conv3)
 ##dense1=AveragePooling1D(pool_size=16,padding='same')(dense1)
 #dense1=Reshape(path1Shape)(dense1)
 
 
 #merge=Concatenate()([dense1,conv3])
 lstm1=Dense(16)(drop) #batch_input_shape=(None,largest,features)#128chr
 lstm1=LayerNormalization(axis=-1)(lstm1)
 #lstm1=Dense(6,activation='relu')(lstm1)
 
 output=Dense(classes,activation='softmax',input_shape=(lstm1.shape))(lstm1)
 model= Model(inputs=input1,outputs=output)
 model.summary()
 return model