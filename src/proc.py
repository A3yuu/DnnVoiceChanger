import numpy as np
from scipy import signal, interpolate

#���K��
def zscore(x, axis = None):
	xmean = x.mean(axis=axis, keepdims=True)
	xstd  = np.std(x, axis=axis, keepdims=True)
	zscore = (x-xmean)/xstd
	return zscore

#�Z���Ԑ����t�[���G
def makeStftWindow(dataLen, waveLengthList):
	stftX = 2*np.pi * dataLen / np.array(waveLengthList)
	stftX = np.array([np.linspace(0, x, dataLen) for x in stftX])
	stftX = np.exp(-stftX*1j)
	stftX *= np.hamming(dataLen)
	return stftX

#�t�[���G�v�Z
def stft(data, stftX):
	return (stftX*data).sum(axis=1)

#�P�g�����o��
def getWave(data, waveLengthList, stftX, order=11, rate=0.5):
	#FT
	ft = stft(data, stftX)
	ftAbsolute = np.abs(ft)
	#�ŏ��̃s�[�N���o��
	mean = max(ftAbsolute)*rate
	ids = signal.argrelmax(ftAbsolute, order=order) #�ɑ�
	ids = ids[0]
	waveLength = -1
	for id in ids:
		if ftAbsolute[id] > mean:
			waveLength = id
	#�G���[����
	if waveLength == -1:
		return (0,0)
	#�ʑ��v�Z
	phase = (np.angle(ft[waveLength])/(np.pi*2) + 1)%1-0.5
	#����
	start = int(0-phase*waveLengthList[waveLength])
	return (start,waveLengthList[waveLength])

#�P�g�����H
def procWave(data, dataPoint, dataBufLen, dataLen, waveLengthList, stftX, fft = True):
	#���o��
	start,orgLen = getWave(data[dataPoint:dataPoint+dataBufLen], waveLengthList, stftX)
	wave = data[dataPoint + start : dataPoint + start + orgLen]
	if orgLen==0 or dataPoint + start < 0:
		return (None, orgLen, None, None)
	#���ʎ擾
	waveMean = np.mean(wave)
	waveStd  = np.std(wave)
	#�������H
	if fft:
		wave = np.fft.fft(wave)
		wave = wave[0:dataLen]
	else:
		t = np.linspace(0, 1, orgLen)
		t2 = np.linspace(0, 1, dataLen)
		wave = interpolate.interp1d(t, wave, kind="quadratic")(t2)
	#�ԋp
	return (zscore(wave), orgLen, waveMean, waveStd)
#�P�g����
def inverseWave(data, orgLen, waveMean, waveStd, octave = 1, normalize = False, fft = True):
	if fft:
		data = np.fft.ifft(data)
		data = np.real(data)
	#���ʖ߂�
	if normalize:
		data = zscore(data)
	data = data * waveStd + waveMean
	#��������
	dataTmp = data
	for i in range(1, octave):
		dataTmp = np.append(dataTmp, data)
	data = dataTmp
	#��Ƃ̒�����
	t = np.linspace(0, 1, len(data))
	t2 = np.linspace(0, 1, orgLen)
	data = interpolate.interp1d(t, data, kind="quadratic")(t2)
	return data

#�G������
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