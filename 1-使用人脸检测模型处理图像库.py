# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
from yolo import YOLO
from PIL import Image


def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def process_image(yolo_, s_file, d_file):
    image = cv2.imread(s_file)
    image = Image.fromarray(image)
    try:
        bbox = yolo_.get_bboxs(image)
        if bbox is not None:
            x1 = int(bbox["left"])
            y1 = int(bbox["top"])
            x2 = int(bbox["right"])
            y2 = int(bbox["bottom"])
            image = np.array(image, dtype='float32')
            image = image[y1:y2 + 1, x1:x2 + 1, :]
            image = cv2.resize(image, (96, 112))
            cv2.imwrite(d_file, image)
            # cv2.imencode('.jpg', image)[1].tofile(d_file)

    except Exception as ex:
        print('-------------------error-------------------------')
        print(ex)


def get_face_dataset(yolo_, s_path, face_dir):
    # 创建face_dir的目录
    for dir in os.listdir(s_path):
        dir_path = face_dir + '/' + dir
        create_path(dir_path)

    for root, dirs, names in os.walk(s_path):
        for filename in names:
            s_file = s_path + '/' + root.split('\\')[-1] + '/' + filename
            d_file = face_dir + '/' + root.split('\\')[-1] + '/' + filename
            process_image(yolo_, s_file, d_file)
            # print(d_file)

    print('done')


if __name__ == '__main__':
    # process_dataset(os.path.join('data', 'dataset'), os.path.join('data', 'result'))
    # 创建目录

    face_dir = 'data/face_dataset'
    create_path(face_dir)

    get_face_dataset(YOLO(), 'data/upload', face_dir)
