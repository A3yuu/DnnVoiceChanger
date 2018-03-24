import numpy as np
from scipy import signal, interpolate
from itertools import chain

from tensorflow.python.keras.models import load_model

import pyaudio

from proc import *

#データ
dataLen = 256
dataBufLen = 1024
audioBufLen = 20000
waveCount = 0
octave = 1
fft = True
#FFT
minWaveLen = 200
maxWaveLen = 450
stepWaveLen = 2
#雑音除去
stackFT = 16
waveQue = []
waveLenQue = []
compFT = 8

#オーディオ設定
audioFormat = pyaudio.paInt16
audioCh = 1        #モノラル
audioRate = 44100        #サンプルレート
audioBuf = 10000       #データ点数

#フーリエ窓関数定義
waveLengthList = range(minWaveLen, maxWaveLen ,stepWaveLen)
stftX = makeStftWindow(dataBufLen, waveLengthList)

#処理
dataBuf =[]
lock = False
dataPoint = 0
def callback(dataRaw, frame_count, time_info, status):
	global dataBuf
	global lock
	global dataPoint
	#バッファリング
	data = np.frombuffer(dataRaw, dtype = "int16")
	while lock:
		pass
	lock = True
	dataBuf.extend(data)
	delLen = len(dataBuf)-audioBufLen
	if delLen > 0:
		dataPoint -= delLen
		if dataPoint < 200:
			dataPoint = 200
		del dataBuf[0:delLen]
	lock = False
	return(None, pyaudio.paContinue)

# モデルを読み込む
model = load_model('model0w256F.h5py')
# 学習結果を読み込む
model.load_weights('model0w256F.weights')
model.summary();
model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

#audio
audio = pyaudio.PyAudio()
stream = audio.open(format=audioFormat, channels=audioCh, rate=audioRate, output=True)
audio.open(format=audioFormat, channels=audioCh, rate=audioRate, input=True, frames_per_buffer=audioBuf, stream_callback=callback)

waveQue = []
while len(dataBuf) != audioBufLen:
	pass
while True:
	#必要長さ貯まるまで待つ
	if dataPoint+dataBufLen > len(dataBuf):
		continue
	while lock:
		pass
	lock = True
	#バッファ取り出し
	procData = list(dataBuf)
	dataPointNow = dataPoint
	lock = False
	#音程
	wave, orgLen, waveMean, waveStd = procWave(procData, dataPointNow, dataBufLen, dataLen, waveLengthList, stftX, n=waveCount, fft=fft)
	if wave is None:
		print("miss")
		dataPoint += 100
		continue
	print("ok")
	dataPoint += orgLen
	#DNN変換
	#waveDnn = np.empty([1,dataLen,2]) #Fourier
	#waveDnn[:,:,0]=wave.real #Fourier
	#waveDnn[:,:,1]=wave.imag #Fourier
	#waveDnn = np.array(wave).real.reshape((1,dataLen,1)) #realのみ
	waveDnn = np.array(wave).reshape((1,dataLen,1)) #ノーマル
	waveDnn = model.predict(waveDnn, verbose=0)
	wave = waveDnn.reshape((dataLen)) #ノーマル & realのみ
	#wave.real = waveDnn[:,:,0] #Fourier
	#wave.imag = waveDnn[:,:,1] #Fourier
	#フーリエ逆変換
	wave = inverseWave(wave, orgLen, waveMean, waveStd, octave = octave, normalize = True, fft=fft)
	#音質改善しない
	wave = np.array(wave).astype(np.int16)
	data = np.frombuffer(np.array(wave), dtype = "uint8")
	stream.write(data)
	continue
	#音質改善
	waveQue.extend(wave)
	waveLenQue.append(len(wave))
	if len(waveLenQue) >= stackFT:
		del waveLenQue[0:len(waveLenQue)-stackFT]
		del waveQue[0:len(waveQue)-sum(waveLenQue)]
		wave = qualityImprovement(waveQue, waveLenQue, compFT, dataLen*compFT, waveStd)
		stackP = stackFT//2//compFT
		start = int(sum(waveLenQue[0:stackP-1]))//compFT
		wave = wave[start:start + waveLenQue[stackP]]
		#再生
		wave = np.array(wave).astype(np.int16)
		data = np.frombuffer(np.array(wave), dtype = "uint8")
		stream.write(data)
