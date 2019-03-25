# Emiya Engine 

> "真正重要的東西，只用眼睛是看不見的。"  
> "只要蘊藏著想成為真物的意志，偽物就比真物還要來得真實。"  

Emiya Engine 是一个用来丰富音频频谱的脚本。可以将频谱变得好看那么一点。  

---

### 当前版本：

`RC Version 2`

### 编年史：

 - `Alpha.0 Rev.3`
 > 这算是 Emiya Engine 的第一个阶段成果，目标的最小实现  
 > 简单说原理就是矩形窗暴力 FFT，移频，乘以乱数，叠加，IFFT  
 > 大部分的代码是为了处理超大数组拼接速度缓慢的问题  
 > 处理后的音频有大量爆音及咔哒声，低电平音频容易看出处理痕迹  
 - `Alpha.1 Rev.0`
 > 该版本为 Alpha.0 的重构改进，主要工作是改写为多进程执行  
 > 为消除频谱图上可见的断层，加入了整数倍时域重采样机制  
 > 事实上重采样带来的运算增加远超多进程带来的提升，所以...  
 > 以及因为多进程，处理需要占用更大的内存，性能消耗巨大  
 > 爆音和咔哒声依旧存在，但已大幅减少，处理痕迹依旧能看出来  
 - `Alpha.1 Rev.1`
 > 这一版中加入了 AkkoMode  
 > 这一模式原理极其简单，就是给原始信号采样点分别乘以极小的随机数  
 > 可以视作信号在有微小热噪声的线路走了一趟  
 > 处理后无爆音及咔哒声，但在低电平音频上能听出背景噪声  
 > 消除背景噪声就必须暴露处理痕迹  
 - `Alpha.3`
 > 推翻了之前的所有代码的完全重构，处理结果类似 DSEE HX   
 > 这一版本质是高通滤波器 + 混频器  
 > 将高通滤波后的信号分离为打击乐及弦乐，然后增益后叠加在原始信号上  
 > 丢掉了自造的 FFT 轮子，改用库实现的 SFFT  
 > 因此不存在爆音和咔哒声，也不再需要额外多倍重采样，速度极大提升  
 > 这一版本参数调节极其重要，需要参照结果反复调整参数  
 > 正确调整参数的处理样本完全不增加爆音及咔哒声，加上 EQ 能完全抹平处理痕迹
 - `RC 1`
 > 加入了参数辅助调整选项  
 > 尝试在 CopyBand 模式下保护动态范围  
 - `RC 2`
 > 加入了参数自动加载/保存特性  
 > 加入了截止/调制频率的优化建议  
 > 提高了系数精度  
 
### RC 版本使用说明：

为了方便使用，特地做了个 GUI 界面，  
但实际上还是挺难用的，所以还是说一下。

工具： Spek（仅频谱观察用），Audition（频谱观察/频率分析/后期处理用）

---

首先需要分析音乐类型，对于以下类型不建议使用 AkkoMode：
 - 电子合成纯音乐，背景乐器只有一两样的
 - 人声清唱带一个伴奏乐器
 - 其他频谱图中最高频率不到 18kHz 的音乐

例如这样的：  
![sample-not-for-akkomode](https://i.imgur.com/Fd4EoGN.jpg?1)

AkkoMode 适用于大部分时候音量都很大的流行乐（比如 JPOP），  
处理时应选用 Apple iTunes 购买的 AAC 格式音频，常见的频谱长这样：  
![some-jpop](https://i.imgur.com/swdtDz6.jpg)  
因为参数只有俩，多试几次就知道，此处就不展开说了。

---

CopyBand 辅助调整：

CopyBand 模式需要设置六个参数，上手困难，  
因此在 `RC 1` 版本中加入了辅助调节手段。  

首先勾选 `优化建议` 和 `样本输出模式` 并将谐波增益倍率设定为 0。然后执行一次。  
执行完成后，将提示窗中的 `建议增益` 填写到冲击增益倍率，再执行一次。  
以此反复，当提示 `建议维持冲击增益` 时，即辅助调节冲击增益完成。  
若需要更好品质，可尝试将谐波增益倍率改为 1，然后再执行一次  
加入谐波后，提示窗中的 `建议增益` 失去参考价值，  
此时可以参考提示的 `当前增益` 进行调节，  
同样的，当提示 `建议维持冲击增益` 时，即调节完成。  
若需要更好的频谱「品相」，可以参考下步骤手动调整。

---

CopyBand 完全手动调整步骤：

配置之前观察频谱。  
以某网站下载的音乐为例，以下是其频谱图及频率分析图：  
![sample-spec-0](https://i.imgur.com/RzEzmtl.jpg) 
![sample-spec-1](https://i.imgur.com/t0ps5iS.png)

从两张图中可以明显看出频率在 17kHz 不到的地方戛然而止，  
如果目标是生成 48kHz 文件，则需要补齐 24-17=7kHz 的部分。  
而 17-7=10kHz，故 HPF 截止频率应设定在 10kHz 以下，  
而调制频率则在 HPF 截止频率上加上 7k。  
本例中设定为 9k 及 16k。  
这首歌背景音乐以打击乐为主，因此能量集中在冲击部分，  
调参数时，首先将谐波增益设置为 0，可以避免参数过多干扰测试。  
冲击增益可以从 5 开始测试，勾上测试模式，启动输出，  
检查输出文件频率分析结果：
![sample-result-0](https://i.imgur.com/gqBmSFy.png)

很明显，在 17-21kHz 的地方本应该是比 17kHz 以下的部分“矮”一些的。  
（高频衰减更大，所以高频部分通常增益应低于低频）  
因此，根据观察结果，将冲击增益调为 2.5（折半试错），再重新跑一次。  
（此时要在其他软件中关闭文件，否则会发生错误）  
调整后的频率分析结果变成了这样：  
![sample-result-1](https://i.imgur.com/aDempeR.png)

此时已经很接近理想的样子了，因为还要加入谐波的部分（前边设定成了 0）  
故再将冲击增益降低 0.5，同时给谐波增益改为 1.0 并再次执行。  
结果变成了这样：  
![sample-result-2](https://i.imgur.com/cAqZdbQ.png)
看起来不错，直接取消测试模式生成最终结果。  
生成最终结果时可能会很卡，请不要担心并耐心等待，进度条将滚动四次（两声道音频）。

接着检查频谱，输出如下：  
![sample-result-3](https://i.imgur.com/E5I1fMf.jpg)  
![sample-result-4](https://i.imgur.com/pFZDbHB.png)  
此时是不是有点失望了，很明显的衔接痕迹对不对。

没关系，这时可以打开 Audition 效果中的 FFT 滤波器，  
接着拿起刚才的频率分析结果图，照着图调整 FFT 滤波器，比如这样：  
![au-fft-filter](https://i.imgur.com/dbHxIKH.png)  
应用后，频率分析结果变成了这样：  
![final-0](https://i.imgur.com/9eYJs8V.png)
而频谱中的衔接痕迹已经不明显了：  
![final-1](https://i.imgur.com/X1cDgcX.jpg)

放大频谱细节，可以看出雾蒙蒙的部分依然有欠缺，  
![final-2](https://i.imgur.com/9AbW9j2.jpg)  
这是谐波增益不够的原因，可以继续调整改善。最终得到以下结果：  
![final-3](https://i.imgur.com/2UO9OnW.jpg)  

### 其他提示：

由于 CopyBand 本质是复制粘贴已有的部分，  
因此对于超过 48kHz 以上的拉升，需要多次处理达成，  
例如以下原始文件不到 16kHz：  
![ex-0](https://i.imgur.com/eAui0i7.jpg)  
拉升到 48kHz 采样需要的频率片段至少为 24-16=8kHz，  
而拉升到 96kHz 采样则需要 48-16=32kHz。  
而原始音频中都没有 32kHz 的容量，  
因此在最终拉升到 96kHz 之前需要重复至少三次操作。  
在这一过程中，最大频率由 16 最终变为 48Hz。  
由于事实上 20kHz 以上的听不见，所以你做得再多也无妨（笑）。  
例如上边的例子被拉升到 192kHz 采样率，宛如天籁之声：  
![ex-1](https://i.imgur.com/QWKpaHA.jpg)

### 特别提醒  
~~请不要使用这个脚本制造 `'HiRes'` 逗玄学家玩~~

