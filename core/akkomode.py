import random
import librosa
import resampy
#from tqdm import tqdm


def core(
    input_path,output_path,
    output_sr=48000,inter_sr=1,
    test_mode=False,
    sv_l=0.02,sv_h=0.55,
    update=None,msgbox=None
):

    # 加载音频
    y, sr = librosa.load(input_path,mono=False,sr=None)
    if test_mode:
        y, sr = librosa.load(input_path,mono=False,sr=None,offset=round(len(y[0])/sr/2),duration=5)
    y = resampy.resample(y, sr, output_sr * inter_sr, filter='kaiser_fast')

    # AkkoMode
    for chan in y:
        # 是否第一次执行
        is_loop_once = True
        # 前一次的数值
        pre_value = 0
        # 前一次操作的数值
        pre_opt = 0
        # 实际操作
        #for i in tqdm(range(len(chan)),unit='Segment',ascii=True):
        for i in range(len(chan)):
            update.emit(i/len(chan))
            this_value = chan[i]
            # 构造抖动值
            linear_jitter = 0
            if pre_value < this_value:
                linear_jitter = random.uniform(this_value*-sv_l, this_value*sv_h)
            else:
                linear_jitter = random.uniform(this_value*sv_h, this_value*-sv_l)
            # 应用抖动
            if pre_opt*linear_jitter > 0:
                chan[i] = this_value + linear_jitter
            elif pre_opt*linear_jitter < 0:
                chan[i] = this_value - linear_jitter
            else:
                pass
            # 第一次操作特殊化处理
            if is_loop_once:
                linear_jitter = random.uniform(this_value*-sv_h, this_value*sv_h)
                chan[i] += linear_jitter
                is_loop_once = False
            # 保存到上一次记录
            pre_value = this_value
            pre_opt = linear_jitter

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
