import numpy as np
import scipy.signal as signal
import librosa
import resampy
#from tqdm import tqdm


def core(
    input_path,output_path,
    output_sr=48000,inter_sr=1,
    test_mode=False,
    harmonic_hpfc=6000,harmonic_sft=16000,harmonic_gain=1.2,
    percussive_hpfc=6000,percussive_stf=16000,percussive_gain=2.5,
    update=None
):

    def hpd_n_shift(data, lpf, sft, gain):
        # 高通滤波
        b,a = signal.butter(3,lpf/(sr/2),'high')
        data = librosa.stft(signal.filtfilt(b,a,librosa.istft(data)))
        # 拷贝频谱
        #for i in tqdm(range(data.shape[1]),unit='Segment',ascii=True):
        for i in range(data.shape[1]):
            update.emit(i/data.shape[1])
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
    y, sr = librosa.load(input_path,mono=False,sr=None)
    if test_mode:
        y, sr = librosa.load(input_path,mono=False,sr=None,offset=round(len(y[0])/sr/2),duration=5)
    y = resampy.resample(y, sr, output_sr * inter_sr, filter='kaiser_fast')
    # 产生 STFT 谱
    stft_list = [librosa.stft(chan) for chan in y]

    # 谐波增强模式
    for chan in stft_list:
        D_harmonic,D_percussive = librosa.decompose.hpss(chan, margin=4)
        D_harmonic = hpd_n_shift(D_harmonic,harmonic_hpfc,harmonic_sft,harmonic_gain)
        D_percussive = hpd_n_shift(D_percussive,percussive_hpfc,percussive_stf,percussive_gain)
        chan += D_harmonic
        chan += D_percussive

    # 合并输出
    istft_list = [librosa.istft(chan) for chan in stft_list]
    final_data = resampy.resample(np.array(istft_list), 
                                  output_sr * inter_sr, 
                                  output_sr, 
                                  filter='kaiser_fast')
    librosa.output.write_wav(output_path, final_data, output_sr)
