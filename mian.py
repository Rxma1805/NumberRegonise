import librosa#.display
# import librosa.feature
import matplotlib.pyplot as plt
from random import shuffle
import os
import numpy as np
from keras.layers import Dense,Input,LSTM,Bidirectional,Activation,Conv1D,GRU
from keras.layers import Embedding,Dropout
import pandas as pd
from keras.callbacks import Callback
from keras.models import Sequential,Model
from keras.optimizers import Adam
import keras.backend as K

dim_input = 44
sequence_len = 128
learning_rate = 5e-5
min_learning_rate = 1e-5
batch_size = 50
step = 1
training_iters = 100000
batch_size = 128
display_step = 10
n_hidden = 350
n_steps=128
file_path = './spoken_numbers_pcm/'

files = os.listdir(file_path)    
labels=[]
mfcc_list=[]
for file in files:
    if not file.endswith(".wav"): continue
    wave, sr = librosa.load(file_path+file, mono=True)
    mfcc = librosa.feature.mfcc(wave, sr,n_mfcc=dim_input,S=librosa.feature.melspectrogram(y=wave, sr=sr, n_mels=8000, fmax=256))           
    label = dense_to_one_hot(int(file.split('_')[0]),10)
    labels.append(label)
    # print(np.array(mfcc).shape)
    mfcc = np.pad(mfcc,((0,0),(0,n_steps-len(mfcc[0]))), mode='constant', constant_values=0)
    mfcc_list.append(np.array(mfcc).T)
np.save('traindata_x.np',np.array(mfcc_list))
np.save('traindata_y.np',np.array(labels))

test_path = './extreme/'
files = os.listdir(test_path)   
test_features = []
test_labels = []

for wav in files:                       
    if not wav.endswith(".wav"): continue
    wave, sr = librosa.load('./extreme/'+wav, mono=True)
    #     label=dense_to_one_hot(int(wav[0]),10)
    test_labels.append(int(wav[0]))
    mfcc = librosa.feature.mfcc(wave, sr,n_mfcc=44,S=librosa.feature.melspectrogram(y=wave, sr=sr, n_mels=8000, fmax=256))
    mfcc=np.pad(mfcc,((0,0),(0,n_steps-len(mfcc[0]))), mode='constant', constant_values=0)
    test_features.append(np.array(mfcc).T)

        
np.save('testdata_x',np.array(test_features))
np.save('testdata_y',np.array(test_labels))

train_data_x = np.load('traindata_x.np.npy')
train_data_y = np.load('traindata_y.np.npy')

test_data_x = np.load('testdata_x.npy')
test_data_y = np.load('testdata_y.npy')

train_data_x.shape,train_data_y.shape,test_data_x.shape,test_data_y.shape

class data_generator:
    def __init__(self, data,label, batch_size):
        self.data = data
        self.label = label
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
    def __len__(self):
        return self.steps
    def __iter__(self):
        global step
        while True:   
            train_x=[]
            train_y=[]
            ids = np.arange(len(self.data))
            print(len(ids))
            shuffle(ids)            
            for i in ids:
                train_x.append(self.data[i])
                train_y.append(self.label[i])
                if len(train_x) == self.batch_size or i == ids[-1]:
#                     print('count:'+str(step))
                    step+=1
                    train_x =  np.array(train_x)
                    train_y = np.array(train_y)
                    yield (train_x,train_y)
                    train_x=[]
                    train_y=[]
                    
train_data = data_generator(train_data_x,train_data_y,batch_size)
def softmax(x):
    x = x - np.max(x)
    x = np.exp(x)
    return x / np.sum(x)

class Evaluate(Callback):
    def __init__(self):        
        self.ACC = []
        self.best = 0.
        self.passed = 0
        
    def on_batch_begin(self,batch,logs=None):
        """第一个epoch用来warmup，第二个epoch把学习率降到最低
        """
        if self.passed < self.params['steps']:
            lr = (self.passed + 1.) / self.params['steps'] * learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1
        elif self.params['steps'] <= self.passed < self.params['steps'] * 2:
            lr = (2 - (self.passed + 1.) / self.params['steps']) * (learning_rate - min_learning_rate)
            lr += min_learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1
    def on_epoch_end(self, epoch, logs=None):
        acc = self.evaluate()
        self.ACC.append(acc)
        if acc > self.best:
            self.best = acc
            model.save_weights('best_model.weights')
        print ('acc: %.4f, best acc: %.4f\n' % (acc, self.best))
    def evaluate(self):
        A = 1e-10
        Y = []
        _result = softmax(model.predict(test_data_x)).argmax(axis=1)
#         print(_result.shape)
        for i,class_label in enumerate(_result):
#             print(class_label)
#             print(test_data_y[i])
            if(class_label == test_data_y[i]):
                A+=1    
#                 print(str(class_label))
        return A / len(_result)
evaluater = Evaluate()
# model.add(Input(shape = (128,)))
model = Sequential()
model.add(Conv1D(filters = 64,kernel_size = 5,strides = 2,padding='same',input_shape=(n_steps,dim_input)))
model.add(Bidirectional(LSTM(n_hidden,return_sequences=True, dropout=0.25, recurrent_dropout=0.1)))
model.add(Bidirectional(LSTM(n_hidden,return_sequences=False, dropout=0.25, recurrent_dropout=0.1)))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(70))
model.add(Activation('relu'))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(10,activation = 'softmax'))
Adam_optimzer = Adam()
# model.build((128,128,44))
model.compile(loss='categorical_crossentropy',optimizer=Adam_optimzer)
print(model.summary())
model.fit_generator(train_data.__iter__(), steps_per_epoch=len(train_data),epochs=100,callbacks=[evaluater])
