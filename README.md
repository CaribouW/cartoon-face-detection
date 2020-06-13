https://blog.csdn.net/keith_bb/article/details/70408907

https://blog.csdn.net/qq_38441692/article/details/88780516?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.nonecase&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.nonecase

https://blog.csdn.net/qq_26898461/article/details/49514787
```
-data : 存放xml文件的目录

-vec : 正样本vec文件源

-bg : 负样本路径txt文件

-numPos : 正样本数量

-numNeg : 负样本数量

numStages：训练分类器的级数

-featureType： 默认使用Haar特征，还有LBP和HOG可供选择(HOG为opencv2.4.0版本以上)

-w -h : 样本宽高

-minHitRate ：分类器的每一级希望得到最小检测率(即正样本被判断有效的比例)

-maxFalseAlarmRate：分类器的每一级希望的最大误检率（负样本判定为正样本的概率）

-mode： 选择训练中使用的Haar特征类型。BASIC只使用右上特征，ALL使用所有右上特征及45度旋转特征(使用Haar特征的前提下，否则不使用此参数)

```


正样本：只含有目标的局部图（若是全图，则需要把目标截取出来，比如训练人脸，则把人脸从含有人脸的图片中截取出来，尺寸要一致），且背景不要太过复杂，灰度图

正样本大小：20\*20(一般用于Haar特征)，==24\*24==（LBP特征）

正样本数量：一般大于等于2000

负样本：不含目标的任何图片，灰度图

负样本大小：60*60

负样本数量：一般大于等于5000

```
nohup make train > log.txt 2>&1 &
```





### 数据集处理

- 首先针对原始图片进行灰度图处理
- 针对每一个灰度图，根据真值 x ,y , w ,h ，得到 $w\times h$ 大小的人脸框，作为正样本集合
- 针对每一个灰度图，根据真值，生成IOU为 0.001的图片，作为负样本集合。每一张图生成大约10 - 20张
- 设置opencv参数, 进行训练