## 卡通人脸检测

### 使用指南

```
.
├── README.md							
├── __init__.py
├── __pycache__
├── bbox_transform.py					基于train.txt生成适用于opencv的文本
├── cascade-39								级联分类器xml文件备份
├── config.py									全局配置
├── data											级联分类器xml文件
├── dataset										训练集/测试集
├── detector.py								基于已经训练好的级联xml文件，对单个图片进行预测，并且绘制bbox
├── env.sh										设置conda环境
├── evaluater.py							模型预测与评估
├── gen_cartoon_middle.py			正样本/负样本扩充
├── gen_middle_dataset				针对真值, 对训练集进行bbox绘制.主要是用于预览使用
├── log.py										日志管理
├── log.txt										日志文件
├── makefile									主要的opencv训练参数在makefile内进行指定
├── middle_data.sh
├── middle_data_set						中间生成的数据集目录
├── output										detector.py产生的图片建议的生成目录
├── pos.vec										opencv生成的向量数据集
├── run.sh										训练脚本,主要由makefile控制参数
├── test.txt									*实验最终的产出结果
├── train.txt									训练集的bbox描述文件
└── utils.py									工具内容
```

当前项目使用opencv3.4.7版本，需要保证在 `~/opencv/build/bin` 目录下存在其样本生成和级联分类器训练的可执行文件

注：如下所有指令均在根目录下执行

- 使用 `python gen_cartoon_middle.py` 来生成中间样本(正样本/负样本)
- 使用 `make gen_sa mple ` 生成 pos.vec 文件
- 使用 `make train` 来进行opencv级联分类器训练
- 使用 `make evaluate` 进行 `test.txt` 的生成。当前可以直接运行这一步，并且使用 `tail -f log.txt` 来查看日志输出
  - 注：当前test.txt中，如果某一个图片没有检测出卡通人脸，不进行空行处理。即文件中不存在空行