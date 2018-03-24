# coding: UTF-8

#GPU使う設定
#import plaidml.keras
#plaidml.keras.install_backend()

import numpy as np

import tensorflow as tf
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import *

from proc import *

#データ
dataPathIn = ['dataX0w64F.npy']
dataPathOut = ['dataY0w64F.npy']
modelPath = ['model0w64F.h5py', 'model0w64F.weights']
dataLen = 64

#学習モデル構築
for path in dataPathIn:
	if 'inData' in locals():
		inData = np.append(inData, np.load(path), axis=0)
	else:
		inData = np.load(path)
for path in dataPathOut:
	if 'outData' in locals():
		outData = np.append(outData, np.load(path), axis=0)
	else:
		outData = np.load(path)
print(inData)
print(outData)
#データ飛ばし
#inData = inData[0::100]
#outData = outData[0::100]
#inData = inData[0:len(inData)//30]
#outData = outData[0:len(outData)//30]
#二次元化
#dim = 2
#R = inData.real
#I = inData.imag
#inData = np.empty((len(inData),dataLen,2))
#inData[:,:,0] = R
#inData[:,:,1] = I
#R = outData.real
#I = outData.imag
#outData = np.empty((len(outData),dataLen,2))
#outData[:,:,0] = R
#outData[:,:,1] = I
#大きさのみ
#dim = 1
#inData = inData.real.reshape((len(inData),dataLen,1))
#outData = outData.real.reshape((len(outData),dataLen,1))
#時間領域
dim = 1
inData = inData.reshape((len(inData),dataLen,1))
#outData = outData.reshape((len(outData),dataLen,1))
#定数
input = Input(shape=(dataLen,dim))
#DenseNet
#blocks = 3
#filters = 16
#growth = 4
#denseLayers = 4
#pooling = 8
#kernelSize = 7
#def DenseBlock(x, layers, filters, growth, kernel):
#	features = [x]
#	for i in range(layers):
#		x = Convolution1D(filters, kernel, padding="same", use_bias=False)(x)
#		features.append(x)
#		x = concatenate(features)
#		filters += growth
#	return x, filters
#def DenseNet(x, blocks, pooling, layers, filters, growth, kernel):
#	for i in range(blocks):
#		x, filters = DenseBlock(x, layers, filters, growth, kernel)
#		x = LeakyReLU(alpha=0.1)(x)
#		#x = BatchNormalization()(x)
#		x = MaxPooling1D(pooling)(x)
#	x = Conv1D(pooling**blocks, 1, kernel_initializer='he_normal')(x)
#	x = Activation('linear')(x) #tanh,linear
#	x = Flatten()(x)
#	#x = Dense(dataLen)(x)
#	#x = BatchNormalization()(x)
#	return x
#x = DenseNet(input, blocks, pooling, denseLayers, filters, growth, kernelSize)
#CNN
blocks = 3
filters = 128
growth = 2
pooling = 4
kernelSize = 7
#blocks = 4
#filters = 64
#growth = 2
#pooling = 4
#kernelSize = 7
x = input
for i in range(blocks):
	x = Conv1D(filters*(growth**i), kernelSize, padding='same', use_bias=False, activation="relu")(x)
	#x = Conv1D(filters*(growth*(i+1)), kernelSize, padding='same', use_bias=False, activation="relu")(x)
	x = BatchNormalization()(x)
	#x = LeakyReLU(alpha=0.1)(x)
	x = MaxPooling1D(pooling)(x)
	#x = AveragePooling1D(pooling)(x)
x = Conv1D(pooling**blocks, 1, kernel_initializer='he_normal')(x)
x = Activation('linear')(x)
#x = Conv1D(dataLen, 1, use_bias=False, activation="relu")(x)
#x = MaxPooling1D(dataLen//(pooling**blocks))(x)
x = Flatten()(x)
#Autoencoder
#blocks = 4
#filters = 16
#kernelSize = 7
#pooling = 4
#x = input
#for i in range(blocks):
#	x = Conv1D(filters, kernelSize, padding='same', use_bias=False, activation="relu")(x)
#	x = MaxPooling1D(pooling)(x)
#for i in range(blocks):
#	x = Conv1D(filters, kernelSize, padding='same', use_bias=False, activation="relu")(x)
#	x = UpSampling1D(pooling)(x)
#x = Conv1D(1, 1, padding='same', use_bias=False, activation="relu")(x)
#x = Flatten()(x)
#RUN
model = tf.keras.Model(inputs=input, outputs=x)
model.summary()
model.save(modelPath[0])
#前回重みロード(なければコメントアウト)
#model.load_weights(modelPath[1])

#Run
#model.compile(optimizer=Adamax(lr = 1e-5), loss='mean_squared_error')
#history = model.fit(inData, outData, epochs=1, verbose=1)
model.compile(optimizer=Adamax(lr = 0.001, decay=0), loss='mean_squared_error')
#epochss = [1,3,6,10,10,10,20,20,20,100,100,100,100,100,100,100,100,100]
#epochss = [1,4,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5]
epochss = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
for i in epochss:
	history = model.fit(inData, outData, epochs=i, verbose=1)
	model.save_weights(modelPath[1])
