import os

import cv2
import numpy as np

import utils
import log

cwd = os.getcwd()
# 级联分类器
cascade_path = cwd + '/data/cascade.xml'
# ground_truth 文件
ground_truth_path = cwd + '/train.txt'
# 最终结果生成文件
predict_path = cwd + '/test.txt'

test_image_dir = cwd + '/dataset/cartoon_dataset/cartoon_test'

iou_threshold = 0.7


def predict(input_path):
    face_cascade = cv2.CascadeClassifier('data/cascade.xml')
    img = cv2.imread(input_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
    res = []
    for (x, y, w, h) in faces:
        res.append([x, y, x + w, y + h])
    return res


def prediction(test_img_dir):
    """
    生成predict.txt文件
    :return:
    """
    pathes = []
    for path, dir_list, file_list in os.walk(test_img_dir):
        pathes = [os.path.join(path, x) for x in file_list]
    pathes.sort()  # 文件名升序排列
    log.logger.info('=======Prediction start.......==============')
    line_cnt = len(pathes)
    idx = 0
    # clean up
    os.system('rm -rf {}'.format(predict_path))
    for file_path in pathes:
        short_path = file_path.split('/')[-1]
        predict_bboxes = predict(file_path)  # 当前图片里面所有的预测脸, xmin,ymin,xmax,ymax
        with open(predict_path, 'a') as f:
            for box in predict_bboxes:
                content = ','.join([short_path, *[str(pos) for pos in box]])
                f.write(content + '\n')
        if idx % 100 == 0:
            log.logger.info('finish predict:\t%d/%d' % (idx, line_cnt))
        idx += 1


def evaluation():
    log.logger.info('=======Evaluation start.......==============')
    TP = 0  # 真阳 , 假阳 . 定义一个候选框和任意真实框的 IOU>0.7 时，该候选框为真阳性 ,样本，否则为假阳性样本
    with open(ground_truth_path) as f:
        g_lines = f.readlines()
        ground_truth_cnt = len(g_lines)
    with open(predict_path) as f:
        p_lines = f.readlines()
        predict_cnt = len(p_lines)
    g_idx, p_idx, = 0, 0

    while g_idx < ground_truth_cnt and p_idx < predict_cnt:
        g_line, p_line = g_lines[g_idx], p_lines[p_idx]
        g_img_idx, p_img_idx = int(g_line.split('.')[0]), int(p_line.split('.')[0])
        if g_img_idx < p_img_idx:
            g_idx += 1
        elif g_img_idx > p_img_idx:
            p_idx += 1
        else:
            def get_faces(lines, line_idx):
                begin_img_idx = int(lines[line_idx].split('.')[0])
                cur_img_idx = begin_img_idx
                ans = []
                while begin_img_idx == cur_img_idx:
                    ans.append([int(x) for x in lines[line_idx].split(',')[1:]])
                    line_idx += 1
                    if line_idx < len(lines):
                        cur_img_idx = int(lines[line_idx].split('.')[0])
                    else:
                        break
                return ans

            g_faces, p_faces = get_faces(g_lines, g_idx), get_faces(p_lines, p_idx)
            g_idx += len(g_faces)
            p_idx += len(p_faces)
            for p_face in p_faces:
                for g_face in g_faces:
                    g_face = np.reshape(g_face, (1, -1))
                    iou = utils.iou(p_face, g_face)
                    # 如果有 iou > 0.7
                    if iou >= iou_threshold:
                        TP += 1
    # 计算TP
    log.logger.info('ground truth face count:\t%d\npredict face count:\t\t\t%d' % (ground_truth_cnt, predict_cnt))
    P = TP / predict_cnt  # precision , 检索到的脸个数 / 所有检索到的脸总数
    R = TP / ground_truth_cnt  # recall , 检索到的脸个数 / 系统所有的真值人脸个数
    F1 = (2 * P * R) / (P + R)
    log.logger.info("TP count:\t{}".format(TP))
    log.logger.info("Precision:\t{}".format(P))
    log.logger.info("Recall:\t{}".format(R))
    log.logger.info("F1:\t{}".format(F1))


if __name__ == '__main__':
    prediction(test_image_dir)
    evaluation()
