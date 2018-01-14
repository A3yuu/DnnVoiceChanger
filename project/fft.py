# coding: UTF-8

import numpy as np
from scipy import signal, interpolate
import time

from scipy.io.wavfile import read
from scipy.io.wavfile import write
import sys
sys.path.append('src')
from proc import *

#データ
dataLen = 54
dataBufLen = 1000
#FFT
minWaveLen = 70
maxWaveLen = 500
stepWaveLen = 1
#雑音除去
stackFT = 8
waveQue = []
waveLenQue = []
compFT = 4

#データ読み込み
names = ["あ.wav","い.wav","う.wav","え.wav","お.wav","か.wav","が.wav","き.wav","ぎ.wav","く.wav","ぐ.wav","け.wav","げ.wav","こ.wav","ご.wav","さ.wav","ざ.wav","し.wav","じ.wav","す.wav","ず.wav","せ.wav","ぜ.wav","そ.wav","ぞ.wav","た.wav","だ.wav","ち.wav","つ.wav","て.wav","で.wav","と.wav","ど.wav","な.wav","に.wav","ぬ.wav","ね.wav","の.wav","は.wav","ば.wav","ぱ.wav","ひ.wav","び.wav","ぴ.wav","ふ.wav","ぶ.wav","ぷ.wav","へ.wav","べ.wav","ぺ.wav","ほ.wav","ぼ.wav","ぽ.wav","ま.wav","み.wav","む.wav","め.wav","も.wav","や.wav","ゆ.wav","よ.wav","ら.wav","り.wav","る.wav","れ.wav","ろ.wav","わ.wav","ん.wav"]
#names = ["あ.wav"]
paths = ["data/i/","data/o/","data/c/","data/o2/"]

#フーリエ窓関数定義
waveLengthList = range(minWaveLen, maxWaveLen ,stepWaveLen)
stftX = makeStftWindow(dataBufLen, waveLengthList)
#てすと
for name in names:
	#読み出し
	fs, dataRaw = read(paths[2] + name)
	dataPoint = 0
	print(np.mean(dataRaw))
	print(np.std(dataRaw))
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
		##ここでDNN変換
		#フーリエ逆変換
		wave = inverseWave(wave, orgLen, waveMean, waveStd, octave = 1)
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
			#キュー
			waves.append(wave)
	#データ構築
	waves = np.array([item for sublist in waves for item in sublist])
	print(np.mean(waves))
	print(np.std(waves))
	waves = waves.astype(np.int16)
	write(paths[3] + name,fs, waves)
