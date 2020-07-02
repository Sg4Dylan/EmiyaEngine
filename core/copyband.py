import numpy as np
import scipy.signal as signal
import librosa
import resampy

def core(
    input_path,output_path,
    output_sr=48000,inter_sr=1,
    test_mode=False,opti_mode=True,dyn_protect=True,
    only_envolope=False,auto_opti=True,no_hpf=True,
    harmonic_hpfc=6000,harmonic_sft=16000,harmonic_gain=1.2,
    percussive_hpfc=6000,percussive_stf=16000,percussive_gain=2.5,
    update=None,msgbox=None
):

    def hpd_n_shift(data, lpf, sft, gain):
        sr = output_sr*inter_sr
        # 计算增益
        gain_src = round(2*data.shape[0]*sft/sr)
        gain_dst = round(gain_src+0.025*data.shape[0])
        src_power = np.mean(np.abs(data[gain_src:gain_dst,:]))
        # 高通滤波
        b,a = signal.butter(3,lpf/(sr/2),'high')
        data = librosa.stft(np.asfortranarray(signal.filtfilt(b,a,librosa.istft(data))))
        # 拷贝频谱
        shift = sft
        shift = sft-lpf
        shift_point = round(2*data.shape[0]*shift/sr)
        # 调制
        for i in range(data.shape[1]):
            update.emit(i/data.shape[1])
            data[:,i] = np.roll(data[:,i], shift_point, axis=0)
        now_power = np.mean(np.abs(data[gain_src:gain_dst,:]))
        # 应用增益
        if auto_opti and no_hpf:
            data /= now_power / src_power
        else:
            data *= gain

        return data
    
    # 仅估计截至频率
    if only_envolope:
        edge_freq = envelope_detect(input_path)
    
        # Tips
        tips = '评估结果：\n'
        tips += f'建议截止频率：{edge_freq*0.4:.02f} Hz\n'
        tips += f'建议调制频率：{edge_freq:.02f} Hz'
        msgbox.emit('估计截至频率',tips,1)
        return

    # Dyn Protect Tips
    if dyn_protect:
        msgbox.emit("提示",
                    "动态范围保护特性已启用\n",
                    1)
    
    # 加载音频
    print('[CopyBand] Info: 加载音频中...')
    y, sr = librosa.load(input_path,mono=False,sr=None)
    if test_mode:
        print('[CopyBand] Info: 正在使用样本输出模式')
        y, sr = librosa.load(input_path,mono=False,sr=None,offset=round(len(y[0])/sr/2),duration=5)
    print(f'[CopyBand] Info: 正在将采样率变换至 {output_sr * inter_sr} Hz...')
    y = resampy.resample(y, sr, output_sr * inter_sr, filter='kaiser_fast')
    # 产生 STFT 谱
    print('[CopyBand] Info: STFT 进行中...')
    stft_list = [librosa.stft(np.asfortranarray(chan)) for chan in y]

    # 谐波增强模式
    for chan in stft_list:
        processed = np.empty(chan.shape)
        

        if no_hpf:
            print('[CopyBand] Info: 开始频谱拷贝...')
            D_percussive = np.copy(chan)
            processed = hpd_n_shift(D_percussive,percussive_hpfc,percussive_stf,percussive_gain)
        else:
            print('[CopyBand] Info: 正在执行 HPSS 分解...')
            D_harmonic,D_percussive = librosa.decompose.hpss(chan, margin=4)
            print('[CopyBand] Info: 开始频谱拷贝...')
            D_harmonic = hpd_n_shift(D_harmonic,harmonic_hpfc,harmonic_sft,harmonic_gain)
            D_percussive = hpd_n_shift(D_percussive,percussive_hpfc,percussive_stf,percussive_gain)
            processed = D_harmonic + D_percussive

        if not dyn_protect:
            chan += processed
        else:
            # 动态范围保护
            adp = processed
            adp_power = np.mean(np.abs(adp))
            src_power  = np.mean(np.abs(chan))
            adj_factor = src_power/(adp_power+src_power)
            sum_chan = np.empty(chan.shape)
            sum_chan = (adp*adj_factor)+(chan*adj_factor)
            chan *= 0
            chan += sum_chan

    # 合并输出
    print('[CopyBand] Info: ISTFT 进行中...')
    istft_list = [librosa.istft(chan) for chan in stft_list]
    print(f'[CopyBand] Info: 采样率变换至 {output_sr} Hz...')
    final_data = resampy.resample(np.array(istft_list), 
                                  output_sr * inter_sr, 
                                  output_sr, 
                                  filter='kaiser_fast')
    
    print('[CopyBand] Info: 正在保存处理结果...')
    try:
        librosa.output.write_wav(output_path, np.asfortranarray(final_data), output_sr)
        print('[CopyBand] Info: 保存处理结果完毕')
    except PermissionError:
        msgbox.emit("警告",
                    "无法写入文件，请检查目标路径写入权限" \
                    "以及文件是否已被其他程序开启。",
                    0)
    
    # 参数优化
    if not opti_mode:
        return
    print('[CopyBand] Info: 评估优化建议...')
    optimizer(input_path,final_data,output_sr,
              percussive_hpfc,percussive_stf,
              percussive_gain,msgbox)


def envelope_detect(input_path):
    # 加载音频
    y, sr = librosa.load(input_path,mono=False,sr=None)
    # 产生 STFT 谱
    stft_list = [librosa.stft(np.asfortranarray(chan)) for chan in y]
    # 阈值
    t_gain = 85
    # 频谱边缘
    edge_freq = 0
    
    for chan in stft_list:
        chan_sum = chan[:,0]
        for i in range(1,chan.shape[1]):
            chan_sum = np.add(chan_sum,chan[:,i])
        
        chan_sum = np.abs(chan_sum)
        chan_max = np.max(chan_sum)
        
        # 滑动窗口
        for p in range(len(chan_sum)):
            s_win = 8
            s_mean = 0
            if p+s_win<len(chan_sum):
                s_mean = np.mean(chan_sum[p:p+s_win])
            else:
                s_mean = np.mean(
                            np.append(
                                chan_sum[p:len(chan_sum)],
                                np.repeat(chan_sum[len(chan_sum)-1],p+s_win-len(chan_sum))
                                )
                            )
            if s_mean < (chan_max/(np.exp2((t_gain)/6))):
                edge_freq += p*sr/2/1024
                break
    
    edge_freq /= 2
    if edge_freq > 2500:
        edge_freq -= 3500
    return edge_freq


def optimizer(
    input_path,
    output_audio, output_sr,
    hpf_cut_freq,
    hpf_mod_freq,
    hpf_gain,
    msgbox,
):

    def hpf_gain_calc():
        # 加载音频
        y, sr = output_audio, output_sr
        # 产生 STFT 谱
        stft_list = [librosa.stft(np.asfortranarray(chan)) for chan in y]
        
        # 加载
        l_power = 0
        h_power = 0
        
        for chan in stft_list:
            # 截止频率为 hpf_cut_freq 的 HPF
            b,a = signal.butter(11,hpf_cut_freq/(sr/2),'high')
            l_data = librosa.stft(np.asfortranarray(signal.filtfilt(b,a,librosa.istft(chan))))
            l_power_sum = np.mean(np.abs(l_data.real))
            
            # 截止频率为 hpf_mod_freq 的 HPF
            b,a = signal.butter(11,hpf_mod_freq/(sr/2),'high')
            h_data = librosa.stft(np.asfortranarray(signal.filtfilt(b,a,librosa.istft(chan))))
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
        
        return l_power, h_power, lnh_power, recomm_r, sr
    
    edge_freq = envelope_detect(input_path)
    l_power, h_power, lnh_power, recomm_r, sr = hpf_gain_calc()
    
    # Tips
    tips = '频率设置建议：\n'
    tips += f'建议截止频率：{edge_freq*0.4:.02f} Hz\n'
    tips += f'建议调制频率：{edge_freq:.02f} Hz\n\n'
    tips += '增益设置建议：\n'
    tips += f'来源加和：{l_power:.03f}\n'
    tips += f'目标加和：{h_power:.03f}\n'
    tips += f'当前增益：{lnh_power:.03f} dB\n'
    if lnh_power < 14.5 or lnh_power > 16:
        tips += f'建议增益：{recomm_r:.03f}'
    else:
        tips += '建议维持冲击增益，微调谐波增益！'
    
    msgbox.emit('优化建议',tips,1)
