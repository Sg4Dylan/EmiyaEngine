# -*- coding: utf-8 -*-
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
from colorama import Fore, Back, init
from PyQt5 import QtCore, QtGui, QtWidgets

class EmiyaEngineCore(QtCore.QThread):

    Update = QtCore.pyqtSignal([str, str, int])
    Finish = QtCore.pyqtSignal()

    ReadyFilePath = ''      # 输入文件
    OutputFilePath = ''     # 输出文件
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
    MidPrintProgress = True # 打印进度信息

    def __init__(self,parent,_InputFilePath,_OutputFilePath,_DebugSwitch,_SplitSize,_WindowSwitch):
        super(EmiyaEngineCore, self).__init__(parent)
        # QtCore.QThread.__init__(self,parent)
        self.ReadyFilePath = _InputFilePath
        self.OutputFilePath = _OutputFilePath
        self.SplitSize = _SplitSize
        self.AnalysisWindow = _WindowSwitch
        if _DebugSwitch == 0:
            MidPrintProgress = False
        elif _DebugSwitch == 1:
            MidPrintProgress = True
        elif _DebugSwitch == 2:
            MidPrint = True
        else:
            pass

    def LoadFile(self):
        self.BeforeSignal, self.BeforeSignalSR = librosa.load(self.ReadyFilePath, sr=None, mono=False)
        self.AfterSignalLeft = np.array([()])
        self.AfterSignalRight = np.array([()])
        print("Load signal complete. ChannelCount: %s SampleRate: %s Hz" % (str(len(self.BeforeSignal)), str(self.BeforeSignalSR)))

    def MidUpSRC(self):
        # 重采样loss样本到96K
        print("Please wait for SRC.")
        self.MidSignalSR = 96000
        self.AfterSignalSR = self.MidSignalSR
        if self.MidSRCFalse:
            self.MidSignal = self.BeforeSignal
        else:
            self.MidSignal = resampy.resample(self.BeforeSignal, self.BeforeSignalSR, self.MidSignalSR, filter='kaiser_best')
        print("Signal SRC complete.")

    def MidFindThresholdPoint(self,_MidFFTResultSingle,_FFTPointCount):
        # 鉴定频谱基本参数
        _MidAmpData = abs(_MidFFTResultSingle[range(_FFTPointCount//2)])
        # Step0. 找出基波幅度
        _MidBaseFreqAmp = _MidAmpData.max()
        if self.MidPrint:
            print("Signal max AMP -> %s" % _MidBaseFreqAmp)
        # Step1. 找出接续的阈值
        _MidThresholdHit = 1.0e-11                                     # 方差判定阈值
        _MidThresholdPoint = 0                                         # 最后的阈值点
        _MidFindRange = int((_FFTPointCount/2)-1)                      # 搜索的范围
        _MidStartFindPos = round(2000/(48000/(_FFTPointCount/2)))      # 从2K频点附近开始寻找，加快速度
        _MidStartFlag = True                                           # 循环用的启动Flag
        _MidLoopCount = 0                                              # 循环计数器
        _MidLegalFreq = 22000                                          # 判定结果合法的阈值频率
        _MidForwardFreq = 3000                                         # 前向修正频率
        _MidOrderFreq = 16000                                          # 钦定频率
        # Rev.1: 检查接续点是否符合常理
        while _MidStartFlag or _MidThresholdPoint>round(_MidLegalFreq/(48000/(_FFTPointCount/2))):
            _MidStartFlag = False
            if (_MidThresholdPoint*(48000/(_FFTPointCount/2))) > int(self.BeforeSignalSR/2):
                _MidThresholdHit *= 2
            for i in range(_MidStartFindPos,_MidFindRange):
                if i+5>_MidFindRange:
                    break
                # 计算连续五个采样*3 的方差，与阈值比较，判断频谱消失的位置
                if np.var(_MidAmpData[i:i+4]) < _MidThresholdHit and \
                   np.var(_MidAmpData[i+1:i+5]) < _MidThresholdHit:
                    # 定位到当前位置的前500Hz位置
                    _MidThresholdPoint = i-round(_MidForwardFreq/(48000/(_FFTPointCount/2)))
                    break
            # 错误超过5把就强行钦定频率为18K
            _MidLoopCount += 1
            if _MidLoopCount > 5:
                _MidThresholdPoint = round(_MidOrderFreq/(48000/(_FFTPointCount/2)))
                break
        # 打印函数返回信息
        if self.MidPrint:
            print("Signal threshold point -> %s @ %sHz  Max Amp -> %s" % (_MidThresholdPoint,_MidThresholdPoint*(48000/(_MidFindRange+1)),_MidBaseFreqAmp))
        # _MidThresholdPoint = round(21000/(48000/(_FFTPointCount/2)))
        return _MidBaseFreqAmp, _MidThresholdPoint

    def MidInsertJitter(self,_MidFFTResultDouble,_FFTPointCount,_MidThresholdPoint,_MidBaseFreqAmp):
        # 构造抖动
        if _MidThresholdPoint <= 0:
            return _MidFFTResultDouble
        for i in range(_MidThresholdPoint,_FFTPointCount-_MidThresholdPoint):
            # Rev.0: 调整生成概率，频率越高概率越低
            # Rev.1: 加入幅值判定，幅度越大概率越大
            _GenPossible = abs((_FFTPointCount/2)-i)/((_FFTPointCount/2)-_MidThresholdPoint)*(_MidBaseFreqAmp/0.22)
            if random.randint(0, 1000000) < 800000 * _GenPossible: # 0<=x<=10
                _MidRealValue = abs(_MidFFTResultDouble.real[i])
                _BaseJitterMin = _MidRealValue * 0.5 * (1-_GenPossible)
                _BaseJitterMax = _MidRealValue * 6 * _GenPossible
                _AmpJitterMin = _MidBaseFreqAmp * _MidRealValue * 0.5
                _AmpJitterMax = _MidBaseFreqAmp * _MidRealValue * 2
                _AmpJitterPrefix = -1 if random.randint(0, 100000) < 50000 else 1
                _MiditterPrefix = -1 if random.randint(0, 100000) < 50000 else 1
                _MidDeltaJitterValue = random.uniform(_BaseJitterMin,_BaseJitterMax) + _AmpJitterPrefix * random.uniform(_AmpJitterMin,_AmpJitterMax)
                _MidFFTResultDouble.real[i] += _MiditterPrefix * _MidDeltaJitterValue
        return _MidFFTResultDouble

    def FinSaveFile(self):
        init(autoreset=True)
        OutputFilePath = os.path.abspath(os.path.join(self.ReadyFilePath, os.pardir)) + "\\"
        OutputFileName = OutputFilePath + 'Output_%s.wav' % uuid.uuid4().hex
        librosa.output.write_wav(OutputFileName, self.AfterSignal, self.AfterSignalSR)
        print(Back.GREEN + Fore.WHITE + "SAVE DONE" + Back.BLACK + " Output path -> " + OutputFileName)

    def run(self):
        # 加载文件启动SRC
        self.LoadFile()
        self.MidUpSRC()
        # 初始化彩色命令行
        init(autoreset=True)
        # 两个声道
        for ChannelIndex in range(2):
            # 记录开始时间
            _MidStartTime = datetime.datetime.now()
            # 信号总长度
            _MidSignalLength = len(self.MidSignal[ChannelIndex])
            # FFT分割数量
            _FFTPointCount = 1024 # 至少2048点，避免计算错误
            _MidDivCount = math.floor(_MidSignalLength/_FFTPointCount)
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
            for SamplePointIndex in range(_MidDivCount+1):
                StartPos = SamplePointIndex*_FFTPointCount
                EndPos = SamplePointIndex*_FFTPointCount+_FFTPointCount
                _EachPieceLeft = np.array([()])
                _EachPieceRight = np.array([()])
                for EachFourPiece in range(int(_FFTPointCount/_EachLength)):
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
                    _MidFFTResultDouble = np.fft.fft(_TempSignal,_FFTPointCount)/(_FFTPointCount)
                    _MidFFTResultSingle = np.fft.fft(_TempSignal,_FFTPointCount)/(_FFTPointCount/2)
                    # 获取当前分段最大振幅, 处理阈值点
                    _MidBaseFreqAmp, _MidThresholdPoint = self.MidFindThresholdPoint(_MidFFTResultSingle,_FFTPointCount)
                    # 构造抖动到当前FFT实际值上
                    _MidFFTAfterJitter = self.MidInsertJitter(_MidFFTResultDouble,_FFTPointCount,_MidThresholdPoint,_MidBaseFreqAmp)
                    # 逆变换IFFT
                    _MidTimerDomSignal = np.fft.ifft(_MidFFTAfterJitter,n=_FFTPointCount)
                    # 接续到新信号上
                    _AppendLength = _EachLength
                    if SuffixFlag:
                        _AppendLength = _FFTPointCount-SuffixLength
                    _MidAppendSignal = _MidTimerDomSignal[0:_AppendLength]
                    if self.MidPrint:
                        print("Per each length -> %s" % len(_MidAppendSignal))
                    if ChannelIndex == 0:
                        _EachPieceLeft = np.append(_EachPieceLeft, _MidAppendSignal)
                    else:
                        _EachPieceRight = np.append(_EachPieceRight, _MidAppendSignal)
                    # 及时跳出尾部
                    if SuffixFlag:
                        break
                # 先倒腾到临时数组，倒腾500次给放回大数组
                if _TempAppendCount < self.SplitSize and not SuffixFlag:
                    # 已消耗时间
                    _MidUsedTime = datetime.datetime.now()-_MidStartTime
                    # 估算剩余时间
                    _MidEtaTime = ((datetime.datetime.now()-_MidStartTime)/((SamplePointIndex+1)/_MidDivCount))-_MidUsedTime
                    # 构造显示文本
                    if ChannelIndex == 0:
                        _TempArrayLeft = np.append(_TempArrayLeft, _EachPieceLeft)
                        if self.MidPrintProgress:
                            print("Left channel progress rate -> " + Fore.CYAN + str(SamplePointIndex) + " / " + str(_MidDivCount-1) + Fore.WHITE + " TIME USED -> " + Fore.YELLOW + str(_MidUsedTime) + Fore.WHITE + " ETA -> " + Fore.GREEN + str(_MidEtaTime))
                    else:
                        _TempArrayRight = np.append(_TempArrayRight, _EachPieceRight)
                        if self.MidPrintProgress:
                            print("Right channel progress rate -> " + Fore.CYAN + str(SamplePointIndex) + " / " + str(_MidDivCount-1) + Fore.WHITE + " TIME USED -> " + Fore.YELLOW + str(_MidUsedTime) + Fore.WHITE + " ETA -> " + Fore.GREEN + str(_MidEtaTime))
                    _MidProgressRate = round(100*(SamplePointIndex+1)/_MidDivCount)
                    self.Update.emit(str(_MidUsedTime)[:-5],str(_MidEtaTime)[:-5],_MidProgressRate)
                else:
                    _TempAppendCount = 0
                    if ChannelIndex == 0:
                        self.AfterSignalLeft = np.append(self.AfterSignalLeft, _TempArrayLeft)
                        _TempArrayLeft = np.array([()])
                    else:
                        self.AfterSignalRight = np.append(self.AfterSignalRight, _TempArrayRight)
                        _TempArrayRight = np.array([()])
                # 倒腾计数器
                _TempAppendCount += 1
                if SuffixFlag:
                    break
        self.AfterSignal = np.array([self.AfterSignalLeft.real, self.AfterSignalRight.real])
        self.FinSaveFile()
        self.Finish.emit()


class EmiyaEngineGUI(object):
    
    ''' Emiya Engine GUI ARGS '''
    
    InputFilePath = ''
    OutputFilePath = ''
    IsUseWindow = False
    SplitSize = 500
    IsStarted = False
    SubCore = QtCore.pyqtSignal()
    
    def __init__(self, MainWindow):
        
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(615, 272)
        MainWindow.setMinimumSize(QtCore.QSize(615, 272))
        MainWindow.setMaximumSize(QtCore.QSize(615, 272))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        MainWindow.setFont(font)
        MainWindow.setWindowTitle("Emiya Engine - 只要蘊藏著想成為真物的意志, 偽物就比真物還要來得真實")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        self.Input_Label = QtWidgets.QLabel(self.centralwidget)
        self.Input_Label.setGeometry(QtCore.QRect(20, 20, 101, 16))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(11)
        self.Input_Label.setFont(font)
        self.Input_Label.setObjectName("Input_Label")
        
        self.InputLineBox = QtWidgets.QLineEdit(self.centralwidget)
        self.InputLineBox.setGeometry(QtCore.QRect(130, 20, 401, 21))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        self.InputLineBox.setFont(font)
        self.InputLineBox.setObjectName("InputLineBox")
        
        self.InputButton = QtWidgets.QPushButton(self.centralwidget)
        self.InputButton.setGeometry(QtCore.QRect(540, 20, 51, 21))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(11)
        self.InputButton.setFont(font)
        self.InputButton.setObjectName("InputButton")
        self.InputButton.clicked.connect(self.SetInputFilePath)
        
        self.OutputLineBox = QtWidgets.QLineEdit(self.centralwidget)
        self.OutputLineBox.setGeometry(QtCore.QRect(130, 50, 401, 21))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        self.OutputLineBox.setFont(font)
        self.OutputLineBox.setObjectName("OutputLineBox")
        
        self.OutputButton = QtWidgets.QPushButton(self.centralwidget)
        self.OutputButton.setGeometry(QtCore.QRect(540, 50, 51, 21))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(11)
        self.OutputButton.setFont(font)
        self.OutputButton.setObjectName("OutputButton")
        self.OutputButton.clicked.connect(self.SetOutputFilePath)
        
        self.Output_Label = QtWidgets.QLabel(self.centralwidget)
        self.Output_Label.setGeometry(QtCore.QRect(20, 50, 101, 16))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(11)
        self.Output_Label.setFont(font)
        self.Output_Label.setObjectName("Output_Label")
        
        self.IsUseWindowCheck = QtWidgets.QCheckBox(self.centralwidget)
        self.IsUseWindowCheck.setGeometry(QtCore.QRect(50, 90, 181, 31))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(11)
        self.IsUseWindowCheck.setFont(font)
        self.IsUseWindowCheck.setChecked(False)
        self.IsUseWindowCheck.setObjectName("IsUseWindowCheck")
        self.IsUseWindowCheck.stateChanged.connect(self.SetIsUseWindowCheck)
        
        self.SplitSizeSpin = QtWidgets.QSpinBox(self.centralwidget)
        self.SplitSizeSpin.setGeometry(QtCore.QRect(350, 90, 71, 31))
        self.SplitSizeSpin.setMinimum(100)
        self.SplitSizeSpin.setMaximum(5000)
        self.SplitSizeSpin.setSingleStep(100)
        self.SplitSizeSpin.setProperty("value", 500)
        self.SplitSizeSpin.setObjectName("SplitSizeSpin")
        self.SplitSizeSpin.valueChanged.connect(self.SetSplitSize)
        
        self.SplitSize_Label = QtWidgets.QLabel(self.centralwidget)
        self.SplitSize_Label.setGeometry(QtCore.QRect(250, 90, 91, 31))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(11)
        self.SplitSize_Label.setFont(font)
        self.SplitSize_Label.setObjectName("SplitSize_Label")
        
        self.StartButton = QtWidgets.QPushButton(self.centralwidget)
        self.StartButton.setGeometry(QtCore.QRect(460, 90, 91, 31))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(11)
        self.StartButton.setFont(font)
        self.StartButton.setObjectName("StartButton")
        self.StartButton.clicked.connect(self.RunProcess)
        
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(20, 130, 571, 101))
        self.groupBox.setObjectName("groupBox")
        
        self.UsedTime_Label = QtWidgets.QLabel(self.groupBox)
        self.UsedTime_Label.setGeometry(QtCore.QRect(30, 30, 101, 16))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(11)
        self.UsedTime_Label.setFont(font)
        self.UsedTime_Label.setObjectName("UsedTime_Label")
        
        self.GlobalProgressBar = QtWidgets.QProgressBar(self.groupBox)
        self.GlobalProgressBar.setGeometry(QtCore.QRect(100, 60, 451, 23))
        self.GlobalProgressBar.setProperty("value", 0)
        self.GlobalProgressBar.setObjectName("GlobalProgressBar")
        
        self.GlobalProgressBar_Label = QtWidgets.QLabel(self.groupBox)
        self.GlobalProgressBar_Label.setGeometry(QtCore.QRect(30, 60, 61, 21))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(11)
        self.GlobalProgressBar_Label.setFont(font)
        self.GlobalProgressBar_Label.setObjectName("GlobalProgressBar_Label")
        
        self.UsedTime = QtWidgets.QLabel(self.groupBox)
        self.UsedTime.setGeometry(QtCore.QRect(150, 30, 101, 16))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(11)
        self.UsedTime.setFont(font)
        self.UsedTime.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.UsedTime.setObjectName("UsedTime")
        
        self.EtaTime = QtWidgets.QLabel(self.groupBox)
        self.EtaTime.setGeometry(QtCore.QRect(410, 30, 91, 16))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(11)
        self.EtaTime.setFont(font)
        self.EtaTime.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.EtaTime.setObjectName("EtaTime")
        
        self.EtaTime_Label = QtWidgets.QLabel(self.groupBox)
        self.EtaTime_Label.setGeometry(QtCore.QRect(290, 30, 101, 16))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(11)
        self.EtaTime_Label.setFont(font)
        self.EtaTime_Label.setObjectName("EtaTime_Label")
        
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 615, 22))
        self.menubar.setObjectName("menubar")
        self.menu_F = QtWidgets.QMenu(self.menubar)
        self.menu_F.setObjectName("menu_F")
        MainWindow.setMenuBar(self.menubar)
        self.action_OpenFile = QtWidgets.QAction(MainWindow)
        self.action_OpenFile.setObjectName("action_OpenFile")
        self.action_OpenFile.triggered.connect(self.SetInputFilePath)
        self.action_Exit = QtWidgets.QAction(MainWindow)
        self.action_Exit.setObjectName("action_Exit")
        self.action_Exit.triggered.connect(self.ExitAll)
        self.menu_F.addAction(self.action_OpenFile)
        self.menu_F.addAction(self.action_Exit)
        self.menubar.addAction(self.menu_F.menuAction())

        self.TextedUI(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        

    def TextedUI(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        self.Input_Label.setText(_translate("MainWindow", "输入文件路径："))
        self.InputButton.setText(_translate("MainWindow", "<<<"))
        self.OutputButton.setText(_translate("MainWindow", "<<<"))
        self.Output_Label.setText(_translate("MainWindow", "输出文件路径："))
        self.IsUseWindowCheck.setToolTip(_translate("MainWindow", "汉宁Hann双余弦窗可以使FFT分析中的频谱泄漏更少"))
        self.IsUseWindowCheck.setText(_translate("MainWindow", "分析音频时使用Hann窗"))
        self.SplitSize_Label.setText(_translate("MainWindow", "倒腾区大小："))
        self.StartButton.setText(_translate("MainWindow", "开始处理"))
        self.groupBox.setTitle(_translate("MainWindow", "输出状态"))
        self.UsedTime_Label.setText(_translate("MainWindow", "当前处理耗时："))
        self.GlobalProgressBar_Label.setText(_translate("MainWindow", "当前进度"))
        self.UsedTime.setText(_translate("MainWindow", "00:00:00"))
        self.EtaTime.setText(_translate("MainWindow", "00:00:00"))
        self.EtaTime_Label.setText(_translate("MainWindow", "预计剩余时间："))
        self.menu_F.setTitle(_translate("MainWindow", "文件(&F)"))
        self.action_OpenFile.setText(_translate("MainWindow", "打开(&O)"))
        self.action_Exit.setText(_translate("MainWindow", "退出(&E)"))
        

    def SetInputFilePath(self):
        self.InputFilePath = str(QtWidgets.QFileDialog.getOpenFileName(None, "选择待处理的文件")[0])
        if self.InputFilePath:
            self.InputLineBox.setText(self.InputFilePath)
    
    def SetOutputFilePath(self):
        self.OutputFilePath = str(QtWidgets.QFileDialog.getSaveFileName(None, "设置输出文件的位置及文件名")[0])
        if self.OutputFilePath:
            self.OutputLineBox.setText(self.OutputFilePath)

    def SetIsUseWindowCheck(self):
        self.IsUseWindow = True if self.IsUseWindowCheck.checkState() == 2 else False

    def SetSplitSize(self):
        self.SplitSize = int(self.SplitSizeSpin.value())

    def ExitAll(self):
        QtWidgets.QApplication.quit()

    def RunProcess(self):
        if not self.IsStarted:
            if self.InputFilePath and self.OutputFilePath:
                self.IsStarted = True
                self.StartButton.setText("停止处理")
                
                self.CoreObject=EmiyaEngineCore(None,self.InputFilePath,self.OutputFilePath,1,self.SplitSize,self.IsUseWindow)
                self.CoreObject.Update.connect(self.UpdateState)
                self.CoreObject.Finish.connect(self.DetectEnd)
                self.CoreObject.start()

    def DetectEnd(self):
        self.IsStarted = False
        self.StartButton.setText("开始处理")

    def UpdateState(self,_TimeUsed,_EtaTime,_ProgressRate):
        self.UsedTime.setText(str(_TimeUsed))
        self.EtaTime.setText(str(_EtaTime))
        self.GlobalProgressBar.setValue(_ProgressRate)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = EmiyaEngineGUI(MainWindow)
    # ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

