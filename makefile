# version of opencv : 3.4.7
CV_PATH = ~/opencv/build/bin
ORIGIN_DATA_DIR = dataset/cartoon_dataset/cartoon_train
MIDDLE_DIR = middle_data_set
DATA_DIR = data
VEC_PATH = pos.vec
# 生成的正样本尺寸
W_H_PARAMS = -w 24 -h 24
idx = 0
# 训练flags配置
TRAIN_FLAGS = -numPos 8000 -numNeg 24000  -numStages 15  -featureType LBP ${W_H_PARAMS} -minHitRate 0.995 -maxFalseAlarmRate 0.4 -numThreads 8 -maxWeakCount 500

# TRAIN_FLAGS = -numPos 5000 -numNeg 15000 -numStages 15 -featureType LBP ${W_H_PARAMS} -minHitRate 0.995 -maxFalseAlarmRate 0.45  # 最终结果 F1 = 0.39 , 24 \times 24

clean:
	rm -rf *.log

env:
	chmod +x *.sh
	bash env.sh

neg_trans:
	find ./middle_data_set/negative -iname "*.jpg" > negatives.txt

gen_middle:
	python gen_cartoon_middle.py

# 生成opencv正样本
gen_sample:
	${CV_PATH}/opencv_createsamples -vec ${VEC_PATH} -info ${MIDDLE_DIR}/pos.txt -bg ${MIDDLE_DIR}/neg.txt ${W_H_PARAMS} -num 70000

# 训练, 日志保存在log.txt
train:
	rm -rf ${DATA_DIR}/*
	${CV_PATH}/opencv_traincascade -data ${DATA_DIR} -vec ${VEC_PATH} -bg ${MIDDLE_DIR}/neg.txt ${TRAIN_FLAGS}

evaluate:
	python evaluater.py


total: train evaluate

backup:
	cp -r ${DATA_DIR} cascade_${idx}

log:
	tail -f log.txt