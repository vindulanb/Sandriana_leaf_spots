# Referred
# https://pythonprogramming.net/introduction-deep-learning-python-tensorflow-keras/
# https://pysource.com/2019/02/15/detecting-colors-hsv-color-space-opencv-with-python/

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
import math

DATADIR = "C:/Users/Vindula/Desktop/Create a DL app with flutter/Datasets/sandriana_leaf"
CATEGORIES = ["des1", "des3"]

# Creating dataset

training_data = []

def create_traing_data():
    low_brown = (8, 100, 20)
    high_brown = (16, 255, 200)
    aspect_ratio = 1.33
    img_weight = 252
    img_height = math.ceil(img_weight / aspect_ratio)

    for category in CATEGORIES:
        # paths to dir
        path = os.path.join(DATADIR, category)
        # 0 for category 1; 1 for category 2
        # As an example 0 for category 1 ; 1 for category 2
        class_num = CATEGORIES.index(category)

        for img in os.listdir(path):
            try:
                img = cv2.imread(os.path.join(path, img))
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                brown_mask = cv2.inRange(hsv, low_brown, high_brown)
                brown = cv2.bitwise_and(img, img, mask=brown_mask)
                new_array = cv2.resize(brown, (img_weight, img_height))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass


create_traing_data()