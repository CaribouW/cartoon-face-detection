# 检测器

import os

import cv2
import numpy as np


def Detector(input_path, output_path):
    face_cascade = cv2.CascadeClassifier('data/cascade.xml')
    img = cv2.imread(input_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
    has_face = False
    for (x, y, w, h) in faces:
        has_face = True
        # 在原图像上绘制矩形
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        print(x, y, w, h)
    if has_face: cv2.imwrite(output_path, img)


if __name__ == '__main__':
    # filename = os.getcwd() + '/dataset/cartoon_dataset/cartoon_train/000001.jpg'
    for i in range(100, 1000):
        filename = os.getcwd() + '/dataset/cartoon_dataset/cartoon_train/001{}.jpg'.format(i)
        Detector(filename, os.getcwd() + '/output/{}.jpg'.format(i))
