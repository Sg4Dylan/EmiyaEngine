#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Emiya Engine
# Version: Alpha.1 Rev.0
# Author: Sg4Dylan - <sg4dylan#gmail.com>
# Created: 02/26/2017
# 真正重要的東西，只用眼睛是看不見的，
# 只要蘊藏著想成為真物的意志，偽物就比真物還要來得真實


import numpy as np
import scipy
import librosa
import resampy
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
    mid_process_ponit = 32
    mid_src_point = 2048
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
        
        
        return [signal_piece, index]

    def save_file(self):
        librosa.output.write_wav(self.output_file_path, self.mid_signal_carray, self.input_sr)

if __name__ == "__main__":

    multiprocessing.freeze_support()
    EmiyaEngine("lossless_short.wav","lossless_test.wav")











































































































