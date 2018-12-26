from PyQt5 import QtWidgets,QtCore,QtGui,uic
import json
import sys
from core import copyband, akkomode

Config = json.loads(open('config.json','rb').read())
Ui_MainWindow, QtBaseClass = uic.loadUiType(Config['ui'])

class Core(QtCore.QThread):
    
    Update = QtCore.pyqtSignal([float])
    Finish = QtCore.pyqtSignal()
    
    def __init__(self, parent, mode, **kwargs):
        super(Core, self).__init__(parent)
        self.mode = mode
        self.kwargs = kwargs
    
    def run(self):
        if self.mode == 0:
            copyband.core(
                self.kwargs['input_path'],self.kwargs['output_path'],
                self.kwargs['output_sr'],self.kwargs['inter_sr'],
                self.kwargs['test_mode'],
                self.kwargs['harmonic_hpfc'],self.kwargs['harmonic_sft'],
                self.kwargs['harmonic_gain'],self.kwargs['percussive_hpfc'],
                self.kwargs['percussive_stf'],self.kwargs['percussive_gain'],
                self.Update)
        else:
            akkomode.core(
                self.kwargs['input_path'],self.kwargs['output_path'],
                self.kwargs['output_sr'],self.kwargs['inter_sr'],
                self.kwargs['test_mode'],
                self.kwargs['sv_l'],self.kwargs['sv_h'],
                self.Update)
        self.Finish.emit()

class MainUI(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self._bind_ui_()
        self.lang = json.loads(open('res/lang.json','rb').read())[Config['lang']]
        self.input_path,self.output_path = None, None
        self.is_started = False
        
    def _bind_ui_(self):
        self.selectInputFile.clicked.connect(lambda:self.openfile(False))
        self.selectOutputFile.clicked.connect(lambda:self.openfile(True))
        self.globalExec.clicked.connect(self.start)

    def openfile(self, is_output):
        if is_output:
            self.output_path,_ = QtWidgets.QFileDialog.getSaveFileName(self,self.lang['OutputDialog'],'','Audio files(*.wav)')
            self.outputFilePath.setText(self.output_path)
        else:
            self.input_path = QtWidgets.QFileDialog.getOpenFileName(self,self.lang['InputDialog'],'','Audio files(*.*)')
            self.inputFilePath.setText(self.input_path[0])

    def start(self):
        
        if (not self.input_path) or (not self.output_path):
            QtWidgets.QMessageBox.warning(self,self.lang['MsgBoxW'],self.lang['LackFile'],QtWidgets.QMessageBox.Ok)
            return
        
        mode = 1
        if self.useCopyBand.isChecked():
            mode = 0
        
        if not self.is_started:
            self.CoreObject = Core(
                None, mode,
                input_path=self.inputFilePath.text(),
                output_path=self.outputFilePath.text(),
                output_sr=int(self.commOutputSr.currentText()[:-2]),
                inter_sr=int(self.commInsertSr.currentText()[:-1]),
                test_mode=self.useSampleOutput.isChecked(),
                harmonic_hpfc=int(self.cbHarmonicHpfCutFreq.value()),
                harmonic_sft=int(self.cbHarmonicShiftFreq.value()),
                harmonic_gain=float(self.cbHarmonicGain.value()),
                percussive_hpfc=int(self.cbPercussiveHpfCutFreq.value()),
                percussive_stf=int(self.cbPercussiveShiftFreq.value()),
                percussive_gain=float(self.cbPercussiveGain.value()),
                sv_l=float(self.akkoJitterDownFactor.value()),
                sv_h=float(self.akkoJitterUpFactor.value()))
            self.CoreObject.start()
            self.CoreObject.Update.connect(self.proc_bar_bind)
            self.CoreObject.Finish.connect(self.proc_end_bind)
            self.globalExec.setText(self.lang['ExecBtnTextStop'])
            self.is_started = True
        else:
            self.CoreObject.terminate()
            self.globalExec.setText(self.lang['ExecBtnTextStart'])
            self.progressBar.setValue(0)
            self.is_started = False
    
    def proc_bar_bind(self, rate):
        self.progressBar.setValue(round(rate*100))
    
    def proc_end_bind(self):
        self.is_started = False
        self.progressBar.setValue(0)
        self.globalExec.setText(self.lang['ExecBtnTextStart'])


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    tl = QtCore.QTranslator()
    tl.load(Config[Config['lang']])
    app.installTranslator(tl)
    window = MainUI()
    window.show()
    sys.exit(app.exec_())
