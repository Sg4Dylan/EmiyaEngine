#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Emiya Engine
# Version: Alpha.1 Rev.0
# Author: Sg4Dylan - <sg4dylan#gmail.com>
# Created: 02/26/2017
# 真正重要的東西，只用眼睛是看不見的，
# 只要蘊藏著想成為真物的意志，偽物就比真物還要來得真實


import math
import numpy as np
import scipy
import librosa
import resampy
import random
import multiprocessing


class EmiyaEngine:

    # 输入参数
    input_file_path = ''
    input_sr = 0
    input_signal_array = ''
    # 中间处理参数
    cpu_thread = 0
    mid_signal_larray = ''
    mid_signal_rarray = ''
    mid_signal_carray = ''
    # 各进程处理参数
    proc_mid_sr_rate = 32
    proc_fft_length = 2048
    # 输出用参数
    output_file_path = ''
    output_sr = 0
    split_size = 500

    def __init__(self, input=None, output=None, output_sr=96000, split_size=500):
        # 导入参数
        if not input or not output:
            print("Missing parameters.")
            return
        self.input_file_path = input
        self.output_file_path = output
        self.output_sr = output_sr
        self.split_size = split_size
        # 初始化程序
        self.init_engine()
        # 启动处理
        self.process_leader()

    def init_engine(self):
        # 将输入文件转为 numpy 数组
        self.input_signal_array, self.input_sr = librosa.load(
            self.input_file_path, sr=None, mono=False)
        print("Load signal complete. ChannelCount: %s SampleRate: %s Hz" % (
            str(len(self.input_signal_array)), str(self.input_sr)))
        # 读取 CPU 线程数
        self.cpu_thread = multiprocessing.cpu_count()

    def process_leader(self):
        # 左右声道
        for channel_index in range(2):
            # 生成进程池
            pool = multiprocessing.Pool(processes=self.cpu_thread)
            # 初始化结果字典
            raw_result_dict = {}
            result_dict = {}
            # 分割时间域
            splited_input_signal = np.array_split(self.input_signal_array[channel_index], self.cpu_thread)
            for i in range(self.cpu_thread):
                # 下发分片
                raw_result_dict[i] = pool.apply_async(self.process_core, (splited_input_signal[i], i, ))
            pool.close()
            pool.join()
            # 拼合分片
            temp_array = np.array([()])
            # 理清顺序
            for i in range(self.cpu_thread):
                result_dict[raw_result_dict[i].get()[1]] = raw_result_dict[i].get()[0]
            # 追加分片
            for i in range(self.cpu_thread):
                temp_array = np.append(temp_array, result_dict[i])
            # 合并到各声道
            if channel_index == 0:
                self.mid_signal_larray = temp_array
            else:
                self.mid_signal_rarray = temp_array
        # 拼合左右声道
        self.mid_signal_carray = np.array([self.mid_signal_larray, self.mid_signal_rarray])
        # 保存文件
        self.save_file()

    def process_core(self, signal_piece, index):
        # SRC 预处理
        proc_mid_data = resampy.resample(signal_piece, 
                                         self.input_sr, 
                                         self.input_sr * self.proc_mid_sr_rate, 
                                         filter='kaiser_fast')
        # 加抖动处理
        
        # SRC 后处理
        proc_fin_piece = resampy.resample(proc_mid_data, 
                                          self.input_sr * self.proc_mid_sr_rate, 
                                          self.output_sr, 
                                          filter='kaiser_fast')
        return [proc_fin_piece, index]

    def find_threshold_point(self, input_fft):
        # 鉴定频谱基本参数
        amp_fft = abs(input_fft[range(self.proc_fft_length // 2)])
        # Step0. 找出基波幅度
        base_amp_freq = amp_fft.max()
        # Step1. 找出接续的阈值
        threshold_hit = 1.0e-11                                                      # 方差判定阈值
        fin_threshold_point = 0                                                      # 最后的阈值点
        find_range = int((self.proc_fft_length / 2) - 1)                             # 搜索的范围
        start_find_pos = round(2000 / (self.input_sr / (self.proc_fft_length / 2)))  # 从2K频点附近开始寻找，加快速度
        start_flag = True                                                            # 循环用的启动Flag
        loop_count = 0                                                               # 循环计数器
        legal_freq = (self.input_sr / 2) - 500                                       # 判定结果合法的阈值频率
        forward_freq = 3000                                                          # 前向修正频率
        order_freq = (self.input_sr / 2) - 6000                                      # 钦定频率
        # Rev.1: 检查接续点是否符合常理
        while start_flag or fin_threshold_point > round(legal_freq / (self.input_sr / (self.proc_fft_length / 2))):
            start_flag = False
            if (fin_threshold_point * (self.input_sr / (self.proc_fft_length / 2))) > int(self.input_sr / 2):
                threshold_hit *= 2
            for i in range(start_find_pos, find_range):
                if i + 5 > find_range:
                    break
                # 计算连续五个采样*3 的方差，与阈值比较，判断频谱消失的位置
                if np.var(amp_fft[i:i + 4]) < threshold_hit and \
                   np.var(amp_fft[i + 1:i + 5]) < threshold_hit:
                    # 定位到当前位置的前500Hz位置
                    fin_threshold_point = i - round(forward_freq / (self.input_sr / (self.proc_fft_length / 2)))
                    break
            # 错误超过5把就强行钦定频率
            loop_count += 1
            if loop_count > 5:
                fin_threshold_point = round(
                    order_freq / (self.input_sr / (self.proc_fft_length / 2)))
                break
        return base_amp_freq, fin_threshold_point

    def generate_jitter(self, input_fft, fin_threshold_point, base_amp_freq):
        # 构造抖动
        if fin_threshold_point <= 0:
            return input_fft
        for i in range(fin_threshold_point, self.proc_fft_length - fin_threshold_point):
            # Rev.0: 调整生成概率，频率越高概率越低
            # Rev.1: 加入幅值判定，幅度越大概率越大
            gen_possible = abs((self.proc_fft_length / 2) - i) / ((self.proc_fft_length /
                                                             2) - fin_threshold_point) * (base_amp_freq / 0.22)
            if random.randint(0, 1000000) < 800000 * gen_possible:  # 0<=x<=10
                real_value = abs(input_fft.real[i])
                base_jitter_min = real_value * 0.5 * (1 - gen_possible)
                base_jitter_max = real_value * 6 * gen_possible
                amp_jitter_min = base_amp_freq * real_value * 0.5
                amp_jitter_max = base_amp_freq * real_value * 2
                amp_jitter_prefix = - \
                    1 if random.randint(0, 100000) < 50000 else 1
                jitter_prefix = - \
                    1 if random.randint(0, 100000) < 50000 else 1
                delta_jitter_value = random.uniform(
                    base_jitter_min, base_jitter_max) + amp_jitter_prefix * random.uniform(amp_jitter_min, amp_jitter_max)
                input_fft.real[
                    i] += jitter_prefix * delta_jitter_value
        return input_fft

    def save_file(self):
        librosa.output.write_wav(self.output_file_path, self.mid_signal_carray, self.output_sr)
        

if __name__ == "__main__":

    multiprocessing.freeze_support()
    EmiyaEngine("lossless_short.wav","lossless_test.wav")











































































































