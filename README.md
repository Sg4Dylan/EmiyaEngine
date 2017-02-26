# Emiya Engine 

> "真正重要的東西，只用眼睛是看不見的。"  
> "只要蘊藏著想成為真物的意志，偽物就比真物還要來得真實。"  

Emiya Engine 是一个用来丰富音频频谱的脚本。可以将频谱变得好看那么一点。  
原理是使用 FFT (快速傅立叶变换) 将音频信号采样转到频域，在频域上为空白的频谱加上与时域幅值相称的微小抖动。  

###敬告：  
你当前处于 dev 分支，本分支下的程序还处于测试开发阶段，使用前请三思。  

###当前版本：  
Alpha.1 Rev.0  
~~终于 Alpha.1 了~~  

###接下来做的：  

 - 重构原有程序结构 （进行中）  
 - 改成多进程处理结构（目前还不知道和现有 GUI 会发生怎样的碰撞）  
 
###当前改进思路：  
提高因采样点数不足而下降的 FFT 精度：  
对低采样点数做时域重采样。  

设置灵活的输出采样率：  
处理过程中生成目标采样率。  

多进程处理结构：  
分割时域处理后等待同步合并。  