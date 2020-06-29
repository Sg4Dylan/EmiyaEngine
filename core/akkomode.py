import random
import librosa
import resampy
import numpy as np


def core(
    input_path,output_path,
    output_sr=48000,inter_sr=1,
    test_mode=False,dyn_protect=True,
    sv_l=0.02,sv_h=0.55,
    update=None,msgbox=None
):

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

    # AkkoMode
    processed_channel = 0
    for chan in y:
        # 行为和最初实现略有不同，但结果相似
        jitter_vector = np.random.uniform(low=sv_l, high=sv_h, size=chan.shape)

        if not dyn_protect:
            chan += chan * jitter_vector
        else:
            # 动态范围保护
            adp = chan * jitter_vector
            adp_power = np.mean(np.abs(adp))
            src_power  = np.mean(np.abs(chan))
            src_f = 1-(adp_power/src_power)
            adp += src_f*chan
            chan *= 0
            chan += adp
        
        # 更新进度条
        processed_channel += 1
        update.emit(processed_channel/len(y))

    # 合并输出
    final_data = resampy.resample(y, 
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
