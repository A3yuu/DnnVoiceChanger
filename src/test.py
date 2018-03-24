# coding: UTF-8

import numpy as np
from scipy import signal, interpolate

from scipy.io.wavfile import read
from scipy.io.wavfile import write

import tensorflow as tf

import matplotlib.pyplot as plt

from proc import *

#データ
dataLen = 256
dataBufLen = 1024
waveCount = 0
octave = 1
fft = True
skipLen = 256
#FFT
minWaveLen = 160
maxWaveLen = 320
stepWaveLen = 2

#データ読み込み
dataName = ['ikuto12.wav', 'res.wav']

# モデルを読み込む
model = tf.keras.models.load_model('model0w256F.h5py')
# 学習結果を読み込む
model.load_weights('model0w256F.weights')
model.summary();

#フーリエ窓関数定義
waveLengthList = range(minWaveLen, maxWaveLen ,stepWaveLen)
stftX = makeStftWindow(dataBufLen, waveLengthList)
#処理
#読み出し
fs, dataRaw = read(dataName[0])
dataRaw = dataRaw[len(dataRaw)*1//60:len(dataRaw)*2//60]
#dataRaw = dataRaw[0:len(dataRaw)//10]
dataPoint = 0
waves = []
while True:
	if(dataPoint+dataBufLen >= len(dataRaw)):
		break
	#音程
	wave, orgLen, waveMean, waveStd = procWave(dataRaw, dataPoint, dataBufLen, dataLen, waveLengthList, stftX, n=waveCount, fft=fft)
	if wave is None:
		print("miss")
		dataPoint += skipLen
		continue
	print('ok')
	dataPoint+=orgLen
	#dataPoint+=orgLen//2
	#wave = zscore(wave)
	#DNN変換
	#waveDnn = np.empty([1,dataLen,2])
	#waveDnn[:,:,0]=wave.real
	#waveDnn[:,:,1]=wave.imag
	#waveDnn = np.array(wave).real.reshape((1,dataLen,1))
	waveDnn = np.array(wave).reshape((1,dataLen,1))
	waveDnn = model.predict(waveDnn, verbose=0)
	wave = waveDnn.reshape((dataLen))
	#wave.real = waveDnn[:,:,0]
	#wave.imag = waveDnn[:,:,1]
	#フーリエ逆変換
	wave = inverseWave(wave, orgLen, waveMean, waveStd, octave = octave, normalize = True, fft=fft)
	#スタック
	waves.append(wave)
	#waves.append(wave[orgLen//4:orgLen//4*3])
#データ構築
waves = np.array([item for sublist in waves for item in sublist])
waves = waves.astype(np.int16)
write(dataName[1], fs, waves)
