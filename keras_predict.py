# -*- coding: utf-8 -*-  
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

batch_size = 128
nb_epoch = 5

def load_sex_data():
   x_train=[]
   Y_train=[]
 
   f = open("keras_model/train.txt","r")
   i = 0
   for line in f.readlines():
       line = line.split(",")
       if i>0:
         if line[0] == "0":
             Y_train.append(0)
         else:
             Y_train.append(1)
         del line[0]
         del line[0]
         x_train.append(line)
       i += 1
   x1=np.array(x_train)
   y1=np.array(Y_train)
   f.close()
   return (x1, y1)
def load_age_data():
   x_train=[]
   Y_train=[]

   f = open("keras_model/train.txt","r")
   i = 0
   for line in f.readlines():
       line = line.strip("\n").split(",")
       if i>0:
         Y_train.append(int(float(line[1])/5))
         del line[0]
         del line[0]
         x_train.append(line)
       i += 1
   x1=np.array(x_train)
   y1=np.array(Y_train)
   f.close()
   return (x1, y1)

def predict_sex(X_test):
    (X_train, y_train) = load_sex_data()
    X_train = X_train.reshape(2058, 22)
    X_test = X_test.reshape(1, 22)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, 2)
  
    #分成3层，中间隐层有32个节点
    model = Sequential()
    model.add(Dense(32, input_shape=(22,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=32))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=2))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
    model.fit(X_train, Y_train,batch_size=batch_size, nb_epoch=nb_epoch,verbose=1)
    
    sex_result = model.predict_classes(X_test,batch_size=1,verbose=1)

    return sex_result
def predict_age(X_test):
    (X_train, y_train) = load_age_data()
    X_train = X_train.reshape(2058, 22)
    X_test = X_test.reshape(1, 22)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, 20)
    #分成3层，中间隐层有32个节点
    model = Sequential()
    model.add(Dense(32, input_shape=(22,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=32))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=20))
    model.add(Activation('softmax'))
    
    sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
    model.fit(X_train, Y_train,batch_size=batch_size, nb_epoch=nb_epoch,verbose=1)
    
    age_result = model.predict_classes(X_test,batch_size=1,verbose=1)
    return age_result
