# 全局配置

import os

cwd = os.getcwd()

# 中间数据目存放目录
GEN_DATA_ROOT_DIR = os.getcwd() + '/middle_data_set'

# 训练集img目录
DATASET_IMG_DIR = cwd + '/dataset/cartoon_dataset/gray_train/'
# 训练集bbox文件路径
ANNO_FILE = cwd + '/dataset/cartoon_dataset/train_bbox_gt.txt'
# ==========================================================================================
# 网络训练相关数据
RE_SIZE = 24

THREAD_HOLD = 0.8
BACKGRAND_NEG_NUM = 40  # negative , 每一张原始图片生成的neg数量
FACE_NUM = 5  # positive , 每一张原始图片生成的pos数量
