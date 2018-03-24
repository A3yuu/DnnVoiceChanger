# coding: UTF-8

import numpy as np
from scipy import signal, interpolate
from scipy.io.wavfile import read

from proc import *

#データ
dataPath = ['dataX0w64F.npy','dataY0w64F.npy']
dataLen = 64
dataBufLen = 256
dataStep = 16
inData = []
outData = []
waveCount = 0
octave = 1
fft = True
#FFT
#300,220,130
#150,110.65
minWaveLenIn = 160
maxWaveLenIn = 320
stepWaveLenIn = 1
minWaveLenOut = 80
maxWaveLenOut = 160
stepWaveLenOut = 1

#データ読み込み
inName = 'ikuto12.wav'
outName = 'hana12.wav'

#フーリエ窓関数定義
waveLengthListIn = range(minWaveLenIn, maxWaveLenIn ,stepWaveLenIn)
waveLengthListOut = range(minWaveLenOut, maxWaveLenOut ,stepWaveLenOut)
stftXIn = makeStftWindow(dataBufLen, waveLengthListIn)
stftXOut = makeStftWindow(dataBufLen, waveLengthListOut)
#学習データ作成
dataPoint = 0
#入力データ
fsIn, dataIn = read(inName)
#dataIn = dataIn[len(dataIn)//3*2:len(dataIn)]
#教師データ
fsOut, dataOut = read(outName)
#dataOut = dataOut[len(dataOut)//3*2:len(dataOut)]
#ループ
c = 0
dataFullLen = min(len(dataIn), len(dataOut))-dataBufLen
for dataPoint in range(0, dataFullLen, dataStep):
	if c%1000 == 0:
		print(str(dataPoint) + '/' + str(dataFullLen) + '(' + str(dataPoint/dataFullLen) + ')')
	#入力データ
	waveDnnIn, orgLen, waveMean, waveStd = procWave(dataIn, dataPoint, dataBufLen, dataLen, waveLengthListIn, stftXIn, n=waveCount, fft = fft)
	if waveDnnIn is None:
		print("miss")
		continue
	#教師データ
	waveDnnOut, orgLen, waveMean, waveStd = procWave(dataOut, dataPoint, dataBufLen, dataLen, waveLengthListOut, stftXOut, n=waveCount*octave, fft = fft)
	if waveDnnOut is None:
		print("miss")
		continue
	#スタック
	inData.append(waveDnnIn)
	outData.append(waveDnnOut)
	c += 1

np.save(dataPath[0], np.array(inData, dtype=np.float32))
np.save(dataPath[1], np.array(outData, dtype=np.float32))
