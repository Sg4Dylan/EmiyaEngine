#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Emiya Engine
# Version: Alpha.0 Rev.3
# Author: Sg4Dylan - <sg4dylan#gmail.com>
# Created: 02/07/2017
# 真正重要的東西，只用眼睛是看不見的，
# 只要蘊藏著想成為真物的意志，偽物就比真物還要來得真實

import os
import argparse
import datetime
import uuid
import random
import math
import numpy as np
import scipy
import librosa
import resampy
import logging
from colorama import Fore, Back, init

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [line:%(lineno)d] \
        %(levelname)s %(message)s',
    datefmt='%a, %d %b %Y %H:%M:%S',
    filename='Emiya.log',
    filemode='a+'
)
logger = logging.getLogger("EmiaLog")
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# add the handler to the root logger
logger.addHandler(console)


class EmiyaEngineCore:

    ReadyFilePath = ''      # 输入文件名
    BeforeSignal = ''       # 输入原始信号
    BeforeSignalSR = 0      # 原始信号采样
    MidSignal = ''          # 重采样到96K信号
    MidSignalSR = 0         # 96K
    AfterSignalLeft = ''    # 处理后的左信号
    AfterSignalRight = ''   # 处理后的右信号
    AfterSignal = ''        # 处理后的信号
    AfterSignalSR = 0       # 处理后的采样率
    SplitSize = 0           # 倒腾区大小
    AnalysisWindow = False  # 分析接续点用的单边FFT是否加窗
    MidSRCFalse = False     # SRC开关, 为True时就会取消掉SRC步骤
    MidPrint = False        # 打印细节日志开关
    MidPrintProgress = True  # 打印进度信息

    def __init__(self, _InputFilePath, _DebugSwitch, _SplitSize, _WindowSwitch):
        # 输入样本:
        # 信号被加载为ndarray结构，有两个声道各以一维array形式存在
        self.ReadyFilePath = _InputFilePath
        self.SplitSize = _SplitSize
        if _WindowSwitch == 1:
            self.AnalysisWindow = True
        if _DebugSwitch == 0:
            MidPrintProgress = False
        elif _DebugSwitch == 1:
            MidPrintProgress = True
        elif _DebugSwitch == 2:
            MidPrint = True
        else:
            pass
        self.LoadFile()
        self.MidUpSRC()
        self.ProcessCore()

    def LoadFile(self):
        self.BeforeSignal, self.BeforeSignalSR = librosa.load(
            self.ReadyFilePath, sr=None, mono=False)
        self.AfterSignalLeft = np.array([()])
        self.AfterSignalRight = np.array([()])
        # print("Load signal complete. ChannelCount: %s SampleRate: %s Hz" %
        #       (str(len(self.BeforeSignal)), str(self.BeforeSignalSR)))
        logger.info("Load signal complete. ChannelCount: %s SampleRate: %s Hz" % (
            str(len(self.BeforeSignal)), str(self.BeforeSignalSR)))

    def MidUpSRC(self):
        # 重采样loss样本到96K
        # print("Please wait for SRC.")
        logger.info("Please wait for SRC.")
        self.MidSignalSR = 96000
        self.AfterSignalSR = self.MidSignalSR
        if self.MidSRCFalse:
            self.MidSignal = self.BeforeSignal
        else:
            self.MidSignal = resampy.resample(
                self.BeforeSignal, self.BeforeSignalSR, self.MidSignalSR, filter='kaiser_best')
        # print("Signal SRC complete.")
        logging.info("Signal SRC complete.")

    def MidFindThresholdPoint(self, _MidFFTResultSingle, _FFTPointCount):
        # 鉴定频谱基本参数
        _MidAmpData = abs(_MidFFTResultSingle[range(_FFTPointCount // 2)])
        # Step0. 找出基波幅度
        _MidBaseFreqAmp = _MidAmpData.max()
        if self.MidPrint:
            # print("Signal max AMP -> %s" % _MidBaseFreqAmp)
            logger.debug("Signal max AMP -> %s" % _MidBaseFreqAmp)
        # Step1. 找出接续的阈值
        _MidThresholdHit = 1.0e-11                                     # 方差判定阈值
        _MidThresholdPoint = 0                                         # 最后的阈值点
        _MidFindRange = int((_FFTPointCount / 2) -
                            1)                      # 搜索的范围
        _MidStartFindPos = round(
            2000 / (48000 / (_FFTPointCount / 2)))      # 从2K频点附近开始寻找，加快速度
        _MidStartFlag = True                                           # 循环用的启动Flag
        _MidLoopCount = 0                                              # 循环计数器
        _MidLegalFreq = 22000                                          # 判定结果合法的阈值频率
        _MidForwardFreq = 3000                                         # 前向修正频率
        _MidOrderFreq = 16000                                          # 钦定频率
        # Rev.1: 检查接续点是否符合常理
        while _MidStartFlag or _MidThresholdPoint > round(_MidLegalFreq / (48000 / (_FFTPointCount / 2))):
            _MidStartFlag = False
            if (_MidThresholdPoint * (48000 / (_FFTPointCount / 2))) > int(self.BeforeSignalSR / 2):
                _MidThresholdHit *= 2
            for i in range(_MidStartFindPos, _MidFindRange):
                if i + 5 > _MidFindRange:
                    break
                # 计算连续五个采样*3 的方差，与阈值比较，判断频谱消失的位置
                if np.var(_MidAmpData[i:i + 4]) < _MidThresholdHit and \
                   np.var(_MidAmpData[i + 1:i + 5]) < _MidThresholdHit:
                    # 定位到当前位置的前500Hz位置
                    _MidThresholdPoint = i - \
                        round(_MidForwardFreq / (48000 / (_FFTPointCount / 2)))
                    break
            # 错误超过5把就强行钦定频率为18K
            _MidLoopCount += 1
            if _MidLoopCount > 5:
                _MidThresholdPoint = round(
                    _MidOrderFreq / (48000 / (_FFTPointCount / 2)))
                break
        # 打印函数返回信息
        if self.MidPrint:
            # print("Signal threshold point -> %s @ %sHz  Max Amp -> %s" % (_MidThresholdPoint,
            #                                                               _MidThresholdPoint * (48000 / (_MidFindRange + 1)), _MidBaseFreqAmp))
            logger.debug("Signal threshold point -> %s @ %sHz  Max Amp -> %s" % (_MidThresholdPoint,
                                                                          _MidThresholdPoint * (48000 / (_MidFindRange + 1)), _MidBaseFreqAmp))
        # _MidThresholdPoint = round(21000/(48000/(_FFTPointCount/2)))
        return _MidBaseFreqAmp, _MidThresholdPoint

    def MidInsertJitter(self, _MidFFTResultDouble, _FFTPointCount, _MidThresholdPoint, _MidBaseFreqAmp):
        # 构造抖动
        if _MidThresholdPoint <= 0:
            return _MidFFTResultDouble
        for i in range(_MidThresholdPoint, _FFTPointCount - _MidThresholdPoint):
            # Rev.0: 调整生成概率，频率越高概率越低
            # Rev.1: 加入幅值判定，幅度越大概率越大
            _GenPossible = abs((_FFTPointCount / 2) - i) / ((_FFTPointCount /
                                                             2) - _MidThresholdPoint) * (_MidBaseFreqAmp / 0.22)
            if random.randint(0, 1000000) < 800000 * _GenPossible:  # 0<=x<=10
                _MidRealValue = abs(_MidFFTResultDouble.real[i])
                _BaseJitterMin = _MidRealValue * 0.5 * (1 - _GenPossible)
                _BaseJitterMax = _MidRealValue * 6 * _GenPossible
                _AmpJitterMin = _MidBaseFreqAmp * _MidRealValue * 0.5
                _AmpJitterMax = _MidBaseFreqAmp * _MidRealValue * 2
                _AmpJitterPrefix = - \
                    1 if random.randint(0, 100000) < 50000 else 1
                _MiditterPrefix = - \
                    1 if random.randint(0, 100000) < 50000 else 1
                _MidDeltaJitterValue = random.uniform(
                    _BaseJitterMin, _BaseJitterMax) + _AmpJitterPrefix * random.uniform(_AmpJitterMin, _AmpJitterMax)
                _MidFFTResultDouble.real[
                    i] += _MiditterPrefix * _MidDeltaJitterValue
        return _MidFFTResultDouble

    def FinSaveFile(self):
        init(autoreset=True)
        OutputFilePath = os.path.abspath(
            os.path.join(self.ReadyFilePath, os.pardir)) + "\\"
        OutputFileName = OutputFilePath + 'Output_%s.wav' % uuid.uuid4().hex
        librosa.output.write_wav(
            OutputFileName, self.AfterSignal, self.AfterSignalSR)
        # print(Back.GREEN + Fore.WHITE + "SAVE DONE" +
        #       Back.BLACK + " Output path -> " + OutputFileName)
        logger.info(Back.GREEN + Fore.WHITE + "SAVE DONE" +
                     Back.BLACK + " Output path -> " + OutputFileName)

    def ProcessCore(self):
        # 初始化彩色命令行
        init(autoreset=True)
        # 两个声道
        for ChannelIndex in range(2):
            # 记录开始时间
            _MidStartTime = datetime.datetime.now()
            # 信号总长度
            _MidSignalLength = len(self.MidSignal[ChannelIndex])
            # FFT分割数量
            _FFTPointCount = 1024  # 至少2048点，避免计算错误
            _MidDivCount = math.floor(_MidSignalLength / _FFTPointCount)
            # 实际重叠操作数量 = FFT分割数量 * 分块次数
            _EachLength = 512
            # 补偿标记, 标记有效时, 说明已经到了序列尾部, 因停止继续循环运算
            SuffixFlag = False
            SuffixLength = 0
            # Rev.2: 加入临时数组加速Append, 临时数组每Append操作设定次数就倒腾一次
            _TempArrayLeft = np.array([()])
            _TempArrayRight = np.array([()])
            _TempAppendCount = 0
            # 除了最后一块，每一块都是取计算结果时域的前512点
            for SamplePointIndex in range(_MidDivCount + 1):
                StartPos = SamplePointIndex * _FFTPointCount
                EndPos = SamplePointIndex * _FFTPointCount + _FFTPointCount
                _EachPieceLeft = np.array([()])
                _EachPieceRight = np.array([()])
                for EachFourPiece in range(int(_FFTPointCount / _EachLength)):
                    StartPos += EachFourPiece * _EachLength
                    EndPos += EachFourPiece * _EachLength
                    # 若超出范围, 需将本次计算完整保留接续有效部分(除去补零部分)
                    if EndPos > _MidSignalLength:
                        EndPos = _MidSignalLength
                        SuffixFlag = True
                    _TempSignal = self.MidSignal[ChannelIndex][StartPos:EndPos]
                    # 不足FFT点数的补零
                    while len(_TempSignal) != _FFTPointCount:
                        _TempSignal = np.append(_TempSignal, [0])
                        SuffixLength += 1
                    # 执行FFT运算, 单边谱用于分析, 双边谱用于处理
                    if self.AnalysisWindow:
                        _TempSignal *= scipy.signal.hann(_FFTPointCount, sym=0)
                    _MidFFTResultDouble = np.fft.fft(
                        _TempSignal, _FFTPointCount) / (_FFTPointCount)
                    _MidFFTResultSingle = np.fft.fft(
                        _TempSignal, _FFTPointCount) / (_FFTPointCount / 2)
                    # 获取当前分段最大振幅, 处理阈值点
                    _MidBaseFreqAmp, _MidThresholdPoint = self.MidFindThresholdPoint(
                        _MidFFTResultSingle, _FFTPointCount)
                    # 构造抖动到当前FFT实际值上
                    _MidFFTAfterJitter = self.MidInsertJitter(
                        _MidFFTResultDouble, _FFTPointCount, _MidThresholdPoint, _MidBaseFreqAmp)
                    # 逆变换IFFT
                    _MidTimerDomSignal = np.fft.ifft(
                        _MidFFTAfterJitter, n=_FFTPointCount)
                    # 接续到新信号上
                    _AppendLength = _EachLength
                    if SuffixFlag:
                        _AppendLength = _FFTPointCount - SuffixLength
                    _MidAppendSignal = _MidTimerDomSignal[0:_AppendLength]
                    if self.MidPrint:
                        # print("Per each length -> %s" % len(_MidAppendSignal))
                        logger.debug("Per each length -> %s" % len(_MidAppendSignal))
                    if ChannelIndex == 0:
                        _EachPieceLeft = np.append(
                            _EachPieceLeft, _MidAppendSignal)
                    else:
                        _EachPieceRight = np.append(
                            _EachPieceRight, _MidAppendSignal)
                    # 及时跳出尾部
                    if SuffixFlag:
                        break
                # 先倒腾到临时数组，倒腾500次给放回大数组
                if _TempAppendCount < self.SplitSize and not SuffixFlag:
                    # 已消耗时间
                    _MidUsedTime = datetime.datetime.now() - _MidStartTime
                    # 估算剩余时间
                    _MidEtaTime = ((datetime.datetime.now() - _MidStartTime) /
                                   ((SamplePointIndex + 1) / _MidDivCount)) - _MidUsedTime
                    # 构造显示文本
                    if ChannelIndex == 0:
                        _TempArrayLeft = np.append(
                            _TempArrayLeft, _EachPieceLeft)
                        if self.MidPrintProgress:
                            # print("Left channel progress rate -> " + Fore.CYAN + str(SamplePointIndex) + " / " + str(_MidDivCount - 1) + Fore.WHITE +
                            #       " TIME USED -> " + Fore.YELLOW + str(_MidUsedTime) + Fore.WHITE + " ETA -> " + Fore.GREEN + str(_MidEtaTime))
                            logger.info("Left channel progress rate -> " + Fore.CYAN + str(SamplePointIndex) + " / " + str(_MidDivCount - 1) + Fore.WHITE +
                                  " TIME USED -> " + Fore.YELLOW + str(_MidUsedTime) + Fore.WHITE + " ETA -> " + Fore.GREEN + str(_MidEtaTime))
                    else:
                        _TempArrayRight = np.append(
                            _TempArrayRight, _EachPieceRight)
                        if self.MidPrintProgress:
                            # print("Right channel progress rate -> " + Fore.CYAN + str(SamplePointIndex) + " / " + str(_MidDivCount - 1) + Fore.WHITE +
                            #       " TIME USED -> " + Fore.YELLOW + str(_MidUsedTime) + Fore.WHITE + " ETA -> " + Fore.GREEN + str(_MidEtaTime))
                            logger.info("Right channel progress rate -> " + Fore.CYAN + str(SamplePointIndex) + " / " + str(_MidDivCount - 1) + Fore.WHITE +
                                  " TIME USED -> " + Fore.YELLOW + str(_MidUsedTime) + Fore.WHITE + " ETA -> " + Fore.GREEN + str(_MidEtaTime))
                else:
                    _TempAppendCount = 0
                    if ChannelIndex == 0:
                        self.AfterSignalLeft = np.append(
                            self.AfterSignalLeft, _TempArrayLeft)
                        _TempArrayLeft = np.array([()])
                    else:
                        self.AfterSignalRight = np.append(
                            self.AfterSignalRight, _TempArrayRight)
                        _TempArrayRight = np.array([()])
                # 倒腾计数器
                _TempAppendCount += 1
                if SuffixFlag:
                    break
        self.AfterSignal = np.array(
            [self.AfterSignalLeft.real, self.AfterSignalRight.real])
        self.FinSaveFile()


def Main(_input, _debug, _size, _window):
    Processor = EmiyaEngineCore(_input, _debug, _size, _window)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description='Emiya Engine\n'
                                                 'Version: Alpha.0 Rev.2\n'
                                                 'Author: Sg4Dylan - <sg4dylan#gmail.com>\n'
                                                 '真正重要的東西, 只用眼睛是看不見的, \n'
                                                 '只要蘊藏著想成為真物的意志, 偽物就比真物還要來得真實.')
    parser.add_argument('-i', '--input', help='待处理文件的绝对路径, 同一路径可直接输入文件名. 例如: \n'
                        'Music_ready_test.mp3')
    parser.add_argument('-d', '--debug', help='调试等级设定. 默认 1 级. \n'
                        '设置为 0 时, 只显示任务起始日志; \n'
                        '设置为 1 时, 额外显示进度日志; \n'
                        '设置为 2 时, 额外显示处理细节日志')
    parser.add_argument('-s', '--size', help='倒腾区大小. 默认 500. \n'
                        '使用倒腾区是因为 numpy 做大数组 append 速度远低于小数组, \n'
                        '故加入小数组多倒腾一手, 这个参数就是小数组的尺寸.')
    parser.add_argument('-w', '--window', help='分析用汉宁Hann双余弦窗启用开关. 默认不使用.\n'
                                               '输入 0 代表不使用, 1 代表使用.')

    args = parser.parse_args()
    _input = args.input
    _debug = args.debug
    _size = args.size
    _window = args.window

    if not _input:
        print("缺少输入文件参数，请使用 --help 参考!")
        exit(1)
    if not _debug:
        _debug = 1
    if not _size:
        _size = 500
    if not _window:
        _window = 0

    Main(_input, _debug, _size, _window)
