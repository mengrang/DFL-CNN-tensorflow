# -*- coding:utf-8 -*-
import json
import os
import tensorflow as tf
import random
import cv2
import numpy as np
import math
import time
from skimage import transform
from PIL import Image, ImageEnhance, ImageFilter
from config import *

def cv_imread(filePath):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
    return cv_img
def get_aug_data(split_dir, image_dir, py_dict, norm=True):
    split_path = os.path.join(FLAGS.data_dir, split_dir)
    image_path = os.path.join(split_path, image_dir)   
    disease_class = py_dict['disease_class']
    image_id = py_dict['image_id']
    zero_array = np.zeros(61)
    zero_array[disease_class] = 1
    label = zero_array
    img = cv_imread(os.path.join(image_path, image_id))
    # random_crop
    # img = _random_crop(img)
    # flip
    img = _flip(img)
    # resized
    # img = _resize(img)
    # img = _pad(img)
    # rotation
    img = _rotation(img)
    img = np.asarray(img)
    # color aug
    # img = _color_augment(img)
    if norm:
        img = img / 255.
    return img, label 

def aux_generator(split_dir, json_file, norm=True):
    # path
    split_path = os.path.join(FLAGS.data_dir, split_dir)
    image_path = os.path.join(split_path, 'pad_images')   
    with open(os.path.join(split_path, json_file), 'r', encoding='utf-8') as f:
        py_list = json.load(f)   
    images = []
    labels = []  
    set_size = len(py_list)
    while True:
        images = []
        labels = []
        i = 0
        while i < FLAGS.batch_size:
            random.shuffle(py_list)
            img, label = get_aug_data(split_dir, 'pad_images', py_list[i], norm)
            images.append(img)
            labels.append(label)
            i += 1
        images = np.asarray(images)
        labels = np.asarray(labels)
        yield images, labels

def generator(split_dir, json_file, norm=True):
    return aux_generator(split_dir, json_file, norm)

def _color_augment(image):
    image.flags.writeable = True  # 将数组改为读写模式
    image = Image.fromarray(np.uint8(image))
    # image.show()
    # 亮度增强
    if random.choice([0, 1]):
        enh_bri = ImageEnhance.Brightness(image)
        brightness = random.choice([0.6, 0.8, 1.2, 1.4])
        image = enh_bri.enhance(brightness)
        # image.show()
    # 色度增强
    if random.choice([0, 1]):
        enh_col = ImageEnhance.Color(image)
        color = random.choice([0.6, 0.8, 1.2, 1.4])
        image = enh_col.enhance(color)
        # image.show()
    # 对比度增强
    if random.choice([0, 1]):
        enh_con = ImageEnhance.Contrast(image)
        contrast = random.choice([0.6, 0.8, 1.2, 1.4])
        image = enh_con.enhance(contrast)
        # image.show()
    # 锐度增强
    if random.choice([0, 1]):
        enh_sha = ImageEnhance.Sharpness(image)
        sharpness = random.choice([0.6, 0.8, 1.2, 1.4])
        image = enh_sha.enhance(sharpness)
        image.show()
    # 模糊
    if random.choice([0, 1]):
        image = image.filter(ImageFilter.BLUR)
    image = np.asarray(image)
    return image

def _flip(image):
    if random.choice([0, 1]):
        direct = random.choice([-1, 0, 1])
        image = cv2.flip(image, direct)
    return image

def _resize(image):
    if random.choice([0, 1]):
        h, w = image.shape[:2]
        ratio = random.choice([0.7, 0.8, 0.9, 1.1, 1.2, 1.3])
        resized_h = math.ceil(h * ratio)
        resized_w = math.ceil(w * ratio)
        image = cv2.resize(image, (resized_w, resized_h))
    return image


def _shift(image):
    if random.choice([0, 1]):
        h, w =image.shape[:2]
        shift_h = random.choice([-h/4, h/4, -h/3, h/3])
        shift_w = random.choice([-h/4, h/4, -w/3, w/3])
        # 在矩阵第一行中表示的是[1,0,x],其中x表示图像将向左或向右移动的距离，如果x是正值，则表示向右移动，如果是负值的话，则表示向左移动。 
        # 在矩阵第二行表示的是[0,1,y],其中y表示图像将向上或向下移动的距离，如果y是正值的话，则向下移动，如果是负值的话，则向上移动
        shift_mat = np.float32([[1, 0, shift_w],[0, 1, shift_h]])
        image = cv2.warpAffine(image, shift_mat, (w, h))
    return image

def _random_crop(image):
    if random.choice([0, 1]):
        size = image.shape
        h = size[0]
        w = size[1]
        ratio = random.uniform(0.7, 0.8)
        h_beg = math.floor(random.uniform(0, h * (1 - ratio)))
        w_beg = math.floor(random.uniform(0, w * (1 - ratio)))
        dh = math.floor(h * ratio)
        dw = math.floor(w * ratio)

        img_crop = image[h_beg:h_beg + dh, w_beg:w_beg + dw, :]
        image = cv2.resize(img_crop, (w, h))
    return image

def _rotation(image):
    if random.choice([0, 1]):
        r_angle = random.randint(-90,90)
        h, w = image.shape[:2]
        # 第一个参数旋转中心，第二个参数旋转角度，第三个参数：缩放比例
        M = cv2.getRotationMatrix2D((h / 2, w / 2), r_angle, 1.)
        # 第三个参数：变换后的图像大小
        # w_rot = math.ceil(h*math.sin(r_angle/180*math.pi)+w*math.cos(r_angle/180*math.pi))+2
        # h_rot = math.ceil(h*math.cos(r_angle/180*math.pi)+w*math.sin(r_angle/180*math.pi))+2
        image = cv2.warpAffine(image, M, (w, h))
    return image
def _pad(image):
    # pading
    img_h = image.shape[0]
    img_w = image.shape[1]
    if max(img_h, img_w) >= 448:
        ratio = max(img_h, img_w) / 447
        image = cv2.resize(image, (math.ceil(img_w//ratio),math.ceil(img_h//ratio)))
        img_h = image.shape[0]
        img_w = image.shape[1]
    img_pad = np.zeros((448, 448, 3), dtype=np.float32)
    h_pad_beg = (448 - img_h) // 2
    w_pad_beg = (448 - img_w) // 2
    h_pad_end = h_pad_beg + img_h
    w_pad_end = w_pad_beg + img_w  
    img_pad[h_pad_beg:h_pad_end, w_pad_beg:w_pad_end, :] = image[:, :, :]
    return img_pad
# if __name__=='__main__':
    
