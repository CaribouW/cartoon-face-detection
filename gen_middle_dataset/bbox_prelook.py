# 针对真值进行bbox绘制
import os

import cv2

import config

cwd = os.getcwd()

if __name__ == '__main__':
    output_dir = cwd + '/dataset/cartoon_dataset/bbox/'
    gray_data_dir = cwd + '/dataset/cartoon_dataset/gray_train/'
    anno_file = config.ANNO_FILE
    with open(anno_file) as f:
        lines = f.readlines()

    line_idx, img_idx = 0, 1
    line_total = len(lines)
    while line_idx < line_total:
        img_name = lines[line_idx].strip()
        img_input_full_path = gray_data_dir + img_name
        img_output_full_path = output_dir + '/' + img_name
        img = cv2.imread(img_input_full_path)
        line_idx += 1
        if line_idx >= line_total: break
        face_cnt = int(lines[line_idx].strip())
        line_idx += 1
        # faces
        faces = [lines[line_idx + i].strip() for i in range(face_cnt)]
        line_idx += face_cnt
        # write
        for L in faces:
            x, y, w, h = [int(x) for x in L.split(' ')]
            # 在原图像上绘制矩形
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imwrite(img_output_full_path, img)
    pass
