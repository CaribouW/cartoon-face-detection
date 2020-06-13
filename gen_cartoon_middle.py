import os
import sys

import cv2
import numpy as np
import numpy.random as npr
import utils
import config
import log

sys.path.append(".")


def gen_middle_img():
    """
    生成 pos ,neg 样本
    :return:
    """

    def gen_negative_anno():
        """
        生成负样本anno文件 , 不涉及尺寸
        :return:
        """
        source_dir = os.getcwd() + '/middle_data_set/negative'
        target_path = os.getcwd() + '/middle_data_set/negatives.txt'
        os.system('find {} -iname "*.jpg" > {}'.format(source_dir, target_path))

    BG_NEG_NUM = config.BACKGRAND_NEG_NUM
    FACE_NUM = config.FACE_NUM
    images_dir = config.DATASET_IMG_DIR
    anno_file = config.ANNO_FILE
    out_dir = config.GEN_DATA_ROOT_DIR

    target_size = 128
    save_dir = '{}'.format(out_dir)

    pos_save_dir = save_dir + '/positive'
    neg_save_dir = save_dir + '/negative'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(pos_save_dir):
        os.mkdir(pos_save_dir)
    if not os.path.exists(neg_save_dir):
        os.mkdir(neg_save_dir)

    f1 = open(os.path.join(save_dir, 'pos' + '.txt'), 'a')
    f2 = open(os.path.join(save_dir, 'neg' + '.txt'), 'a')

    with open(anno_file, 'r') as f:
        annotations = f.readlines()

    n_lines = len(annotations)
    log.logger.info('%d pics in total' % n_lines)
    p_idx = 0  # positive
    n_idx = 0  # negative
    idx = 0

    idx_line = 0
    while idx_line < n_lines:
        # 获取图片及bbox
        image_name = annotations[idx_line].strip()

        n_faces = int(annotations[idx_line + 1])

        image_path = os.path.join(images_dir, image_name)
        img = cv2.imread(image_path)

        bboxes = []
        for i in range(n_faces):
            anno = annotations[idx_line + 2 + i].strip().split()
            anno = list(map(int, anno))
            x1, y1, w, h = anno[0], anno[1], anno[2], anno[3]  # 获得基本数据
            box = utils.convert_bbox((x1, y1, w, h), False)  # 存储 x1 , y1 ,x2 ,y2
            bboxes.append(box)
        bboxes = np.array(bboxes, dtype=np.float32)

        idx_line += 1 + n_faces + 1

        idx += 1
        if idx % 1000 == 0:
            log.logger.info(idx, 'images done')
        height, width, channel = img.shape
        # ====================
        # 随机生成 negative 样本
        neg_num = 0
        neg_limit = 0
        while neg_num < BG_NEG_NUM and neg_limit < 1000:
            try:
                size = npr.randint(0, min(width, height) / 2)
                nx = npr.randint(0, width - size)
                ny = npr.randint(0, height - size)
                crop_box = np.array([nx, ny, nx + size, ny + size])

                _iou = utils.iou(crop_box, bboxes)

                cropped_im = img[ny: ny + size, nx: nx + size, :]
                resized_im = cv2.resize(cropped_im, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            except Exception:
                neg_limit += 1
                continue
            # 保证生成的负样本和真值IOU小于 0.01
            if np.max(_iou) < 0.01:
                save_file = os.path.join(neg_save_dir, '%s.jpg' % n_idx)
                f2.write(save_dir + '/negative/%s.jpg' % n_idx + '\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
                neg_num += 1
            else:
                neg_limit += 1
            log.logger.info('{} images done, pos: {}, neg: {}'.format(idx, p_idx, n_idx))
        # ====================
        for box in bboxes:
            x1, y1, x2, y2 = box
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            face_num, limit_cnt, bg_num = 0, 0, 0
            # 生成pos
            while face_num < FACE_NUM and limit_cnt < 100:
                try:
                    size = npr.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))
                    delta_x = npr.randint(-w * 0.1, w * 0.1)
                    delta_y = npr.randint(-h * 0.1, h * 0.1)

                    nx1 = int(max(x1 + w / 2 + delta_x - size / 2, 0))
                    ny1 = int(max(y1 + h / 2 + delta_y - size / 2, 0))
                    nx2 = nx1 + size
                    ny2 = ny1 + size
                    crop_box = np.array([nx1, ny1, nx2, ny2])

                    cropped_im = img[ny1: ny2, nx1: nx2, :]
                    resized_im = cv2.resize(cropped_im, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
                except Exception:
                    limit_cnt += 1
                    continue
                box_ = box.reshape(1, -1)
                _iou = utils.iou(crop_box, box_)
                if max(_iou) >= 0.7:
                    save_file = os.path.join(pos_save_dir, '%s.jpg' % p_idx)
                    f1.write('positive/%s.jpg' % p_idx + ' 1 0 0 %d %d\n' % (target_size - 1, target_size - 1))
                    cv2.imwrite(save_file, resized_im)
                    face_num += 1
                    p_idx += 1
                else:
                    limit_cnt += 1
                if limit_cnt >= 100:
                    cropped_im = img[int(y1): int(y2), int(x1): int(x2), :]
                    save_file = os.path.join(pos_save_dir, '%s.jpg' % p_idx)
                    cv2.imwrite(save_file, cropped_im)
                    f1.write('positive/%s.jpg' % p_idx + ' 1 0 0 %d %d\n' % (w - 1, h - 1))
                    p_idx += 1
            log.logger.info('{} images done, pos: {},  neg: {}'.format(idx, p_idx, n_idx))
    f1.close()
    f2.close()
    gen_negative_anno()  # 使用shell指令生成negative的anno文件


def gen_gray_img():
    """
    为opencv生成对应的pos样本bbox
    :return: 生成训练图像的灰度图
    """

    def img_to_gray(input_path, output_path):
        img_gray = cv2.imread(input_path)
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_RGB2GRAY)
        cv2.imwrite(output_path, img_gray)

    anno_file = config.ANNO_FILE
    origin_data_dir = os.getcwd() + '/dataset/cartoon_dataset/cartoon_train'
    output_dir = os.getcwd() + '/dataset/cartoon_dataset/gray_train'
    output_file = output_dir + '/positives.txt'

    with open(anno_file) as f:
        lines = f.readlines()

    line_idx, img_idx = 0, 1
    line_total = len(lines)

    while line_idx < line_total:
        img_name = lines[line_idx].strip()
        img_to_gray(origin_data_dir + '/' + img_name, output_dir + '/' + img_name)  # 存储灰度图
        line_idx += 1
        if line_idx >= line_total: break
        face_cnt = int(lines[line_idx].strip())
        line_idx += 1
        # faces
        faces = [lines[line_idx + i].strip() for i in range(face_cnt)]
        line_idx += face_cnt
        # write
        with open(output_file, 'a') as f:
            content = '{} {} {}'.format(img_name, str(face_cnt), ' '.join(faces))
            f.write(content + '\n')


if __name__ == '__main__':
    gen_middle_img()
