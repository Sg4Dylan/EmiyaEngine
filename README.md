# Emiya Engine 

> "真正重要的東西，只用眼睛是看不見的。"  
> "只要蘊藏著想成為真物的意志，偽物就比真物還要來得真實。"  

Emiya Engine 是一个用来丰富音频频谱的脚本。可以将频谱变得好看那么一点。  
原理是使用 FFT (快速傅立叶变换) 将音频信号采样转到频域，在频域上为空白的频谱加上与时域幅值相称的微小抖动。

###使用须知：

 -  由于 FFT 的栅栏效应，程序的处理过程不可避免地会损失部分采样信号。故不建议将本程序用于玄学领域。
 -  由于程序缺乏细致调教，当前算法会导致部分采样块未被有效处理。
 -  由于玄学原因，最后生成的文件有一定几率出现时长一百余个采样点（0.0016s）的爆音区域。
 -  由于处理需要消耗大约是音源时长的 2 - 10 倍的时间，故不建议输入较长的音频文件。
 -  鉴于脚本输出为 96KHz 采样 32bit 单精度浮点型 的 WAV 文件，请不要输入比输出精度更高的文件。

###程序依赖：

 - Python 3
 - numpy
 - scipy
 - librosa
 - resampy
 - colorama

####依赖安装建议
Windows平台：

> 实际上 librosa 的依赖非常多，如果使用 `pip`安装，可能会导致出错，  
> 建议直接在[这里](http://www.lfd.uci.edu/~gohlke/pythonlibs/)下载二进制包使用`pip`离线安装。  
> 除了以上列表的依赖，还有`Cython` 和 `scikit-learn`，建议一块装了。  
> `librosa` 需要配置 `ffmpeg` 的目录，  
> 找到 `Python` 安装目录下 `Lib\site-packages\audioread` 文件夹的 `ffdec.py` 文件。  
> 修改第 32 行，修改为你的 `ffmpeg` 程序路径，比如我的放在 F 盘根目录，设置成这样：  
> `COMMANDS = ('F:\\ffmpeg.exe', 'avconv')`  

Linux平台：

> ArchLinux 上建议用 pacman 一路搞定 numpy scipy scikit-learn Cython，当然使用pip也是OK的。  
> Debian/Ubuntu 上默认源似乎必须用 apt-get 安装，有一些包用 pip 安装有些问题。  
> 与 Windows 平台一样，也需要为 librosa安装后端解码音频文件，  
> 直接使用发行版自带的包管理器安装 ffmpeg 就可以。  
> 如果出错了，建议从源编译最新的放在原来的路径下。  

###命令行帮助：

    -h, --help          显示帮助信息.
    -i INPUT, --input INPUT
                        待处理文件的绝对路径, 同一路径可直接输入文件名. 例如:
                        Music_ready_test.mp3
    -d DEBUG, --debug DEBUG
                        调试等级设定. 默认 1 级.
                        设置为 0 时, 只显示任务起始日志;
                        设置为 1 时, 额外显示进度日志;
                        设置为 2 时, 额外显示处理细节日志
    -s SIZE, --size SIZE  倒腾区大小. 默认 500.
                        使用倒腾区是因为 numpy 做大数组 append 速度远低于小数组,
                        故加入小数组多倒腾一手, 这个参数就是小数组的尺寸.

###效果预览：
音源：44.1KHz@16bit WAV
![enter image description here](https://i.imgur.com/VU9Obqw.jpg)

