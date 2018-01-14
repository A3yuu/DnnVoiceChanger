# coding: UTF-8

#GPU使う設定
#import plaidml.keras
#plaidml.keras.install_backend()

import numpy as np
#from pylab import *
from scipy import signal, interpolate
from scipy.io.wavfile import read

import tensorflow as tf
import sys
sys.path.append('src')
from proc import *

#データ
dataLen = 54
dataBufLen = 1000
dataStep = 100
inData = []
outData = []
#FFT
minWaveLen = 70
maxWaveLen = 500
stepWaveLen = 1

#データ読み込み
names = ["あ.wav","い.wav","う.wav","え.wav","お.wav","か.wav","が.wav","き.wav","ぎ.wav","く.wav","ぐ.wav","け.wav","げ.wav","こ.wav","ご.wav","さ.wav","ざ.wav","し.wav","じ.wav","す.wav","ず.wav","せ.wav","ぜ.wav","そ.wav","ぞ.wav","た.wav","だ.wav","ち.wav","つ.wav","て.wav","で.wav","と.wav","ど.wav","な.wav","に.wav","ぬ.wav","ね.wav","の.wav","は.wav","ば.wav","ぱ.wav","ひ.wav","び.wav","ぴ.wav","ふ.wav","ぶ.wav","ぷ.wav","へ.wav","べ.wav","ぺ.wav","ほ.wav","ぼ.wav","ぽ.wav","ま.wav","み.wav","む.wav","め.wav","も.wav","や.wav","ゆ.wav","よ.wav","ら.wav","り.wav","る.wav","れ.wav","ろ.wav","わ.wav","ん.wav"]
#names = ["あ.wav","い.wav","う.wav","え.wav","お.wav"]
#names = ["あ.wav"]
paths = ["data/i/","data/o/","data/c/","data/o2/"]

#フーリエ窓関数定義
waveLengthList = range(minWaveLen, maxWaveLen ,stepWaveLen)
stftX = makeStftWindow(dataBufLen, waveLengthList)
#学習データ作成
for name in names:
	dataPoint = 0
	#入力データ
	fsIn, dataIn = read(paths[0] + name)
	#教師データ
	fsOut, dataOut = read(paths[1] + name)
	#ループ
	for dataPoint in range(0, min(len(dataIn), len(dataOut))-dataBufLen, dataStep):
		#入力データ
		waveDnnIn, orgLen, waveMean, waveStd = procWave(dataIn, dataPoint, dataBufLen, dataLen, waveLengthList, stftX)
		if waveDnnIn is None:
			print("miss")
			continue
		#教師データ
		waveDnnOut, orgLen, waveMean, waveStd = procWave(dataOut, dataPoint, dataBufLen, dataLen, waveLengthList, stftX)
		if waveDnnOut is None:
			print("miss")
			continue
		#スタック
		inData.append(waveDnnIn)
		outData.append(waveDnnOut)

#tftest
#nums = [
#	(64,7),
#	(64,5),
#	(64,3),
#	]
#inData = np.array(inData)
#outData = np.array(outData)
#inData = inData.reshape((len(inData),dataLen,1))
#outData = outData.reshape((len(outData),dataLen,1))
#import tensorflow as tf
#sess = tf.InteractiveSession()
#x = tf.placeholder(tf.float32, shape=[None, dataLen, 1])
#y = tf.placeholder(tf.float32, shape=[None, dataLen, 1])
#f = x
#for num, k in nums:
#	w =tf.Variable(tf.truncated_normal([num, k, 1]), dtype=tf.float32)
#	f = tf.nn.conv1d(f, w, padding='SAME', stride=1)
#nums.reverse()
#for num, k in nums:
#	w =tf.Variable(tf.truncated_normal([num, k, 1]), dtype=tf.float32)
#	f = tf.nn.conv1d(f, w, padding='SAME', stride=1)
#w =tf.Variable(tf.truncated_normal([1, 3, 1]), dtype=tf.float32)
#f = tf.nn.conv1d(f, w, padding='SAME', stride=1)
#loss = tf.reduce_mean(tf.abs(f - y))
#optimizer = tf.train.GradientDescentOptimizer(0.5)
#train = optimizer.minimize(loss)
#sess = tf.Session()
#sess.run(tf.global_variables_initializer())
#for step in range(201):
#	sess.run(train, feed_dict={x: inData, y: outData})
#	print(loss.eval(session=sess, feed_dict={x: inData, y: outData}))
#sess.close()

#学習モデル構築
inData = np.array(inData)
outData = np.array(outData)
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
dim = 1
inData = inData.real.reshape((len(inData),dataLen,1))
outData = outData.real.reshape((len(outData),dataLen,1))
#時間領域
#dim = 1
#inData = inData.reshape((len(inData),dataLen,1))
#outData = outData.reshape((len(outData),dataLen,1))
nums = [
	(64,7),
	(64,5),
	(64,3),
	]
layerTimes = 3
ks = 3 #3 17
input = tf.keras.layers.Input(shape=(dataLen,dim))
x = input
for num, k in nums:
	x = tf.keras.layers.Conv1D(num, k, activation='relu', padding='same')(x)
	x = tf.keras.layers.MaxPooling1D(layerTimes, padding='same')(x)
nums.reverse()
for num, k in nums:
	x = tf.keras.layers.Conv1D(num, k, activation='relu', padding='same')(x)
	x = tf.keras.layers.UpSampling1D(layerTimes)(x)
x = tf.keras.layers.Conv1D(dim, ks, activation='sigmoid', padding='same')(x)
x = tf.keras.layers.BatchNormalization()(x)
model = tf.keras.Model(inputs=input, outputs=x)
model.compile(optimizer='adam', loss='mean_squared_error')

#Run
history = model.fit(inData, outData, epochs=100, verbose=1)
#Save
model.save('model.h5py')
model.save_weights('model.weights');

##学習モデル構築
#dim = 1
#inData = np.array(inData)
#outData = np.array(outData)
#inData = inData.reshape((len(inData),dataLen,1))
#outData = outData.reshape((len(outData),dataLen,1))
#nums = [
#	(64,7),
#	(64,5),
#	(64,3),
#	]
#layerTimes = 4
#ks = 3 #3 17
#model = Sequential()
#model.add(Layer(input_shape=(dataLen,dim)))
##model.add(BatchNormalization(input_shape=(dataLen,dim)))
#for num, k in nums:
#	model.add(Conv1D(num, k, activation='relu', padding='same'))
#	model.add(MaxPooling1D(layerTimes, padding='same'))
#nums.reverse()
#for num, k in nums:
#	model.add(Conv1D(num, k, activation='relu', padding='same'))
#	model.add(UpSampling1D(layerTimes))
#model.add(Conv1D(dim, ks, activation='sigmoid', padding='same'))
##model.add(Dense(1, activation="sigmoid"))
#model.add(BatchNormalization())
#model.compile(optimizer='adam', loss='mean_squared_error')
#
##Run
#history = model.fit(inData, outData, epochs=1000, verbose=1)
##Save
#model.save('model.h5')
#model.save_weights('model.weights');
