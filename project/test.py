# coding: UTF-8

import numpy as np
from scipy import signal, interpolate

from scipy.io.wavfile import read
from scipy.io.wavfile import write

import tensorflow as tf

import matplotlib.pyplot as plt
import sys
sys.path.append('src')
from proc import *

#データ
dataLen = 54
dataBufLen = 1000
#FFT
minWaveLen = 200
maxWaveLen = 450
stepWaveLen = 2

#データ読み込み

names = ["あ.wav","い.wav","う.wav","え.wav","お.wav","か.wav","が.wav","き.wav","ぎ.wav","く.wav","ぐ.wav","け.wav","げ.wav","こ.wav","ご.wav","さ.wav","ざ.wav","し.wav","じ.wav","す.wav","ず.wav","せ.wav","ぜ.wav","そ.wav","ぞ.wav","た.wav","だ.wav","ち.wav","つ.wav","て.wav","で.wav","と.wav","ど.wav","な.wav","に.wav","ぬ.wav","ね.wav","の.wav","は.wav","ば.wav","ぱ.wav","ひ.wav","び.wav","ぴ.wav","ふ.wav","ぶ.wav","ぷ.wav","へ.wav","べ.wav","ぺ.wav","ほ.wav","ぼ.wav","ぽ.wav","ま.wav","み.wav","む.wav","め.wav","も.wav","や.wav","ゆ.wav","よ.wav","ら.wav","り.wav","る.wav","れ.wav","ろ.wav","わ.wav","ん.wav"]
#names = ["あ.wav","い.wav","う.wav","え.wav","お.wav"]
paths = ["data/i/","data/o/","data/c/","data/o2/"]

# モデルを読み込む
model = tf.keras.models.load_model('model.h5py')
# 学習結果を読み込む
model.load_weights('model.weights')
model.summary();
model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

#フーリエ窓関数定義
waveLengthList = range(minWaveLen, maxWaveLen ,stepWaveLen)
stftX = makeStftWindow(dataBufLen, waveLengthList)
#処理
for name in names:
	#読み出し
	fs, dataRaw = read(paths[0] + name)
	dataPoint = 0
	waves = []
	while True:
		if(dataPoint+dataBufLen >= len(dataRaw)):
			break
		#音程
		wave, orgLen, waveMean, waveStd = procWave(dataRaw, dataPoint, dataBufLen, dataLen, waveLengthList, stftX)
		if wave is None:
			print("miss")
			dataPoint += 100
			continue
		dataPoint+=orgLen
		wave = zscore(wave)
		#DNN変換
		#waveDnn = np.empty([1,dataLen,2])
		#waveDnn[:,:,0]=wave.real
		#waveDnn[:,:,1]=wave.imag
		waveDnn = np.array(wave).real.reshape((1,dataLen,1))
		waveDnn = model.predict(waveDnn, verbose=0)
		wave = waveDnn.reshape((dataLen))
		#wave.real = waveDnn[:,:,0]
		#wave.imag = waveDnn[:,:,1]
		#フーリエ逆変換
		wave = inverseWave(wave, orgLen, waveMean, waveStd, octave = 3, normalize = True)
		#スタック
		waves.append(wave)
	#データ構築
	waves = np.array([item for sublist in waves for item in sublist])
	waves = waves.astype(np.int16)
	write(paths[2] + name,fs, waves)
