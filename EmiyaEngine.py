#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Emiya Engine
# Version: Alpha.3 Copyband aka DSEE HX
# Author: Sg4Dylan - <sg4dylan#gmail.com>
# Created: 07/01/2018
# 真正重要的東西，只用眼睛是看不見的，
# 只要蘊藏著想成為真物的意志，偽物就比真物還要來得真實

import numpy as np
import scipy.signal as signal
import librosa
import resampy
from tqdm import tqdm

# 待处理文件位置
file_path = 'Input.mp3'
# 欲输出采样率
output_sr = 48000
# 中间处理采样率倍数
mid_sr_rate = 1
# HPF 截止频率, 调制频率, 增益 
# (请在观察过原始音频频谱后修改，必须手工修改后使用)
harmonic_hpfc,harmonic_sft,harmonic_gain = 3000,4200,0.9
percussive_hpfc,percussive_stf,percussive_gain = 2000,4000,1.5

def hpd_n_shift(data, lpf, sft, gain):
    # 高通滤波
    b,a = signal.butter(3,lpf/(sr/2),'high')
    data = librosa.stft(signal.filtfilt(b,a,librosa.istft(data)))
    # 拷贝频谱
    for i in tqdm(range(data.shape[1]),unit='Segment',ascii=True):
        shift = sft
        shift_point = round(shift/(sr/data.shape[0]))
        # 调制
        for p in reversed(range(len(chan[:,i]))):
            data[:,i][p] = data[:,i][p-shift_point]
    # 高通滤波
    data = librosa.stft(signal.filtfilt(b,a,librosa.istft(data)))
    data *= gain
    return data

# 加载音频
print('Opening file...')
y, sr = librosa.load(file_path,mono=False,sr=None) # offset=40,duration=5,
print('Resampling to HiRes...')
y = resampy.resample(y, sr, output_sr * mid_sr_rate, filter='kaiser_fast')
# 产生 STFT 谱
print('Generating STFT data...')
stft_list = [librosa.stft(chan) for chan in y]
# 显示基本信息
print(f'InputSr: {sr}, OutputSr: {output_sr}, Shape: {stft_list[0].shape}')

# 谐波增强模式
print('Processing...')
for chan in stft_list:
    print('Generating HPSS data...')
    D_harmonic,D_percussive = librosa.decompose.hpss(chan, margin=4)
    print('...')
    D_harmonic = hpd_n_shift(D_harmonic,harmonic_hpfc,harmonic_sft,harmonic_gain)
    D_percussive = hpd_n_shift(D_percussive,percussive_hpfc,percussive_stf,percussive_gain)
    chan += D_harmonic
    chan += D_percussive

# 合并输出
print('Generating output file...')
istft_list = [librosa.istft(chan) for chan in stft_list]
final_data = resampy.resample(np.array(istft_list), 
                              output_sr * mid_sr_rate, 
                              output_sr, 
                              filter='kaiser_fast')
print('Writing wave...')
librosa.output.write_wav('eh_output.wav', final_data, output_sr)
