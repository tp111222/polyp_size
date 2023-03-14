# -*- coding: UTF-8 -*-
import os
import cv2
import numpy as np
from models import engines


class ModelsPredict:
    def __init__(self, camera_param_path):
        self.ModelsPredict = engines.ModelsPredict()
        self.fx, self.fy = self.get_camera_param(camera_param_path)

    def model_load_init(self):
        ret = self.ModelsPredict.init()
        return ret

    def infer(self, img_path):
        outputs = self.ModelsPredict.infer(img_path)
        return outputs

    def get_camera_param(self, camera_param_path):
        mtx = []
        camera_param_txt = os.path.join(camera_param_path)
        if os.path.exists(camera_param_txt):
            files = open(camera_param_txt, 'r')
            files = files.readlines()
            for file in files:
                file = file.strip('\n').strip(' ').split(' ')
                for i in range(len(file)):
                    file[i] = float(file[i])
                mtx.append(file)
        fx, fy = mtx[0][0], mtx[1][1]
        return fx, fy

    def calculate_fx_fy_norm(self, img_w, img_h):
        fx_norm = self.fx / img_w
        fy_norm = self.fy / img_h

        return fx_norm, fy_norm

    def calculate_size(self, img):
        img_h, img_w, _ = img.shape
        xmin, ymin, xmax, ymax, depth_value = self.infer(img)
        bbox_w = (xmax - xmin)/img_w
        bbox_h = (ymax - ymin)/img_h
        fx_norm, fy_norm = self.calculate_fx_fy_norm(img_w, img_h)
        w = depth_value * bbox_w / fx_norm
        h = depth_value * bbox_h / fy_norm
        size = (w + h)/2
        return size, xmin, ymin, xmax, ymax


def display(img, boxes, predict_size, save_img_path):
    predict_size = '{:.1f}'.format(predict_size)
    img_h, img_w, _ = img.shape
    xmin = int(boxes[0])
    ymin = int(boxes[1])
    xmax = int(boxes[2])
    ymax = int(boxes[3])
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)
    cv2.putText(img, str(predict_size) + 'mm', (int(img_w / 2), 100 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (0, 255, 255), 3, cv2.LINE_AA)

    cv2.imwrite(save_img_path, img)


def main():
    camera_param_path = os.path.join('.', 'camera_param', 'camera_param.txt')
    img_path = os.path.join(".", "image", "test.jpg")
    result_dir = os.path.join(".", "result")
    models_predict = ModelsPredict(camera_param_path)
    img = cv2.imread(img_path)
    predict_size, xmin, ymin, xmax, ymax = models_predict.calculate_size(img)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    save_img_path = os.path.join(result_dir, img_path.split(os.sep)[-1])
    boxes = [xmin, ymin, xmax, ymax]
    display(img, boxes, predict_size, save_img_path)


if __name__ == "__main__":
    main()

