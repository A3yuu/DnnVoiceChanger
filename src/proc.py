# coding: UTF-8

import numpy as np
from scipy import signal, interpolate

#import matplotlib.pyplot as plt

#正規化
def zscore(x, axis = None):
	xmean = x.mean(axis=axis, keepdims=True)
	xstd  = np.std(x, axis=axis, keepdims=True)
	zscore = (x-xmean)/xstd
	return zscore
def zscore2(x, axis = None):
	return x/np.std(x, axis=axis, keepdims=True)

#短時間精密フーリエ
def makeStftWindow(dataLen, waveLengthList):
	stftX = 2*np.pi * dataLen / np.array(waveLengthList)
	stftX = np.array([np.linspace(0, x, dataLen) for x in stftX])
	stftX = np.exp(-stftX*1j)
	stftX *= np.hamming(dataLen)
	return stftX

#フーリエ計算
def stft(data, stftX):
	return (stftX*data).sum(axis=1)

#１波長取り出し
def getWave(data, waveLengthList, stftX, order=11, rate=0.5):
	#FT
	ft = stft(data, stftX)
	ftAbsolute = np.abs(ft)
	#plt.plot(ftAbsolute)
	#plt.show()
	#最初のピーク取り出し
	mean = max(ftAbsolute)*rate
	ids = signal.argrelmax(ftAbsolute, order=order) #極大
	ids = ids[0]
	waveLength = -1
	for id in ids:
		if ftAbsolute[id] > mean:
			waveLength = id
	#エラー処理
	if waveLength == -1:
		return (0,0)
	#位相計算
	phase = (np.angle(ft[waveLength])/(np.pi*2) + 1)%1-0.5
	#結果
	start = int(0-phase*waveLengthList[waveLength])
	return (start,waveLengthList[waveLength])

#１波長加工
def procWave(data, dataPoint, dataBufLen, dataLen, waveLengthList, stftX, fft = False, hamming = True, n=1):
	#取り出し
	if n>0:
		start, orgLen = getWave(data[dataPoint:dataPoint+dataBufLen], waveLengthList, stftX)
		orgLen *= n
		if orgLen==0 or dataPoint + start < 0 or dataPoint + start + orgLen > len(data):
			return (None, orgLen, None, None)
	else:
		start, orgLen = 0, dataBufLen
	wave = data[dataPoint + start : dataPoint + start + orgLen]
	#音量取得
	waveMean = np.mean(wave)
	waveStd  = np.std(wave)
	if waveStd < 1:
		return (None, orgLen, None, None)
	#長さ加工
	if fft:
		if hamming:
			wave = wave * np.hamming(orgLen)
		wave = np.fft.fft(wave)
		wave = np.abs(wave[0:dataLen])
		#wave = np.pad(wave,(0,dataLen-len(wave)), 'constant', constant_values=(0))
		#返却
		return (zscore2(wave), orgLen, waveMean, waveStd)
	else:
		t = np.linspace(0, 1, orgLen)
		t2 = np.linspace(0, 1, dataLen)
		wave = interpolate.interp1d(t, wave, kind="quadratic")(t2)
		#返却
		return (zscore(wave), orgLen, waveMean, waveStd)
#１波長解凍
def inverseWave(data, orgLen, waveMean, waveStd, octave = 1, normalize = False, fft = False, hamming = False):
	if fft:
		data = np.fft.ifft(data, n=orgLen)
		data = np.real(data)
		if hamming:
			data *= np.hamming(orgLen)
	#音量戻す
	if normalize:
		data = zscore(data)
	data = data * waveStd + waveMean
	#音程調整
	dataTmp = data
	for i in range(1, octave):
		dataTmp = np.append(dataTmp, data)
	data = dataTmp
	#もとの長さへ
	t = np.linspace(0, 1, len(data))
	t2 = np.linspace(0, 1, orgLen)
	data = interpolate.interp1d(t, data, kind="quadratic")(t2)
	return data

#雑音除去
def qualityImprovement(wave, waveLens, compFT, compLen, waveStd):
	stackFT = len(waveLens)
	stackFT2 = stackFT//2
	orgLen = len(wave)//compFT
	wave = wave * np.hamming(len(wave))
	waveF = np.fft.fft(wave)[0:compLen]
	waveA = np.abs(waveF)
	#waveTh = np.angle(waveF)
	ids = signal.argrelmax(waveA, order=stackFT2)[0]
	waveA2 = np.zeros(compLen)
	#waveTh2 = np.zeros(compLen)
	for id in ids:
		if id//compFT < compLen:
			waveA2[id//compFT] = waveA[id]
			#waveTh2[id//compFT] = waveTh[id]
	#waveF = np.exp(waveTh2*1j)*waveA2
	waveF = waveA2
	wave = np.fft.ifft(waveF, n = orgLen).real
	return zscore(wave) * waveStd