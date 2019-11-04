import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import cv2
import matplotlib.pyplot as plt
from math import sqrt as sqrt

# 需要检查的数据
sets=[('2020', 'train')]

# 需要检查的类别
classes = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15']

if __name__ == '__main__':
    # GT框宽高统计
    width = []
    height = []

    for year, image_set in sets:
        # 图片ID不带后缀
        image_ids = open('VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
        for image_id in image_ids:
            # 图片的路径
            img_path = 'VOC%s/JPEGImages/%s.jpg'%(year, image_id)
            # 这张图片的XML标注路径
            label_file = open('VOC%s/Annotations/%s.xml' % (year, image_id))
            tree = ET.parse(label_file)
            root = tree.getroot()
            size = root.find('size')
            img_w = int(size.find('width').text)
            img_h = int(size.find('height').text)
            img = cv2.imread(img_path)
            for obj in root.iter('object'):
                difficult = obj.find('difficult').text
                cls = obj.find('name').text
                if cls not in classes or int(difficult) == 2:
                    continue
                cls_id = classes.index(cls)

                xmlbox = obj.find('bndbox')
                xmin = float(xmlbox.find('xmin').text)
                ymin = float(xmlbox.find('ymin').text)
                xmax = float(xmlbox.find('xmax').text)
                ymax = float(xmlbox.find('ymax').text)
                w = xmax - xmin
                h = ymax - ymin
                # width.append(w)
                # height.append(h)
                img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 8)
                w_change = (w / img_w) * 416
                h_change = (h / img_h) * 416
                # width.append(w_change)
                # height.append(h_change)
                s = w_change * h_change
                width.append(sqrt(s))
                height.append(w_change / h_change)
            print(img_path)
            img = cv2.resize(img, (608, 608))
            cv2.imshow('result', img)
            cv2.waitKey()

    plt.plot(width, height, 'ro')
    plt.show()