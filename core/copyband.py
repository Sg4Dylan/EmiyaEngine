import numpy as np
import scipy.signal as signal
import librosa
import resampy


def core(
    input_path,output_path,
    output_sr=48000,inter_sr=1,
    test_mode=False,opti_mode=True,dyn_protect=True,
    harmonic_hpfc=6000,harmonic_sft=16000,harmonic_gain=1.2,
    percussive_hpfc=6000,percussive_stf=16000,percussive_gain=2.5,
    update=None,msgbox=None
):

    def hpd_n_shift(data, lpf, sft, gain):
        # 高通滤波
        b,a = signal.butter(3,lpf/(sr/2),'high')
        data = librosa.stft(signal.filtfilt(b,a,librosa.istft(data)))
        # 拷贝频谱
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
    
    # Dyn Protect Tips
    if dyn_protect:
        msgbox.emit("提示",
                    "动态范围保护特性已启用\n",
                    1)
    
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
        
        if not dyn_protect:
            chan += D_harmonic + D_percussive
        else:
            # 动态范围保护
            adp = D_harmonic + D_percussive
            adp_power = np.mean(np.abs(adp))
            src_power  = np.mean(np.abs(chan))
            src_f = 1-(adp_power/src_power)
            d_ls.append(src_f)
            adp += src_f*chan
            chan *= 0
            chan += adp


    # 合并输出
    istft_list = [librosa.istft(chan) for chan in stft_list]
    final_data = resampy.resample(np.array(istft_list), 
                                  output_sr * inter_sr, 
                                  output_sr, 
                                  filter='kaiser_fast')
    try:
        librosa.output.write_wav(output_path, final_data, output_sr)
    except PermissionError:
        msgbox.emit("警告",
                    "无法写入文件，请检查目标路径写入权限" \
                    "以及文件是否已被其他程序开启。",
                    0)
    # 参数优化
    if not opti_mode:
        return
    optimizer(output_path,
              percussive_hpfc,percussive_stf,
              percussive_gain,msgbox)


def optimizer(
    output_path,
    hpf_cut_freq,
    hpf_mod_freq,
    hpf_gain,
    msgbox
):
    
    # 加载音频
    y, sr = librosa.load(output_path,mono=False,sr=None)
    # 产生 STFT 谱
    stft_list = [librosa.stft(chan) for chan in y]
    
    # 加载
    l_power = 0
    h_power = 0
    for chan in stft_list:
        # 截止频率为 hpf_cut_freq 的 HPF
        b,a = signal.butter(11,hpf_cut_freq/(sr/2),'high')
        l_data = librosa.stft(signal.filtfilt(b,a,librosa.istft(chan)))
        l_power_sum = np.mean(np.abs(l_data.real))
        
        # 截止频率为 hpf_mod_freq 的 HPF
        b,a = signal.butter(11,hpf_mod_freq/(sr/2),'high')
        h_data = librosa.stft(signal.filtfilt(b,a,librosa.istft(chan)))
        h_power_sum = np.mean(np.abs(h_data.real))
        
        # 移相差分
        l_power_sum -= h_power_sum
        
        # 合并音轨数据
        l_power += l_power_sum
        h_power += h_power_sum
        
        # 测试 HPF 输出
        chan *= 0
        chan += h_data
    
    # 频带比例矫正能量比例
    pf = ((sr/2)-hpf_mod_freq)/(hpf_mod_freq-hpf_cut_freq)
    l_power *= pf
    # 计算当前增益比率
    lnh_power = np.log2(l_power/h_power) * 6
    # 如果是正数，就是增益偏小
    target_r  = np.exp2((lnh_power-14.8)/6)
    recomm_r = target_r*hpf_gain
    # Tips
    tips = ''
    tips += f'来源加和：{l_power}\n'
    tips += f'目标加和：{h_power}\n'
    tips += f'当前增益：{lnh_power} dB\n'
    if lnh_power < 14.5 or lnh_power > 16:
        tips += f'建议增益：{recomm_r}'
    else:
        tips += '建议维持冲击增益，微调谐波增益！'
    
    msgbox.emit('优化建议',tips,1)
    #istft_list = [librosa.istft(chan) for chan in stft_list]
    #librosa.output.write_wav('kk.wav', np.array(istft_list), sr)
