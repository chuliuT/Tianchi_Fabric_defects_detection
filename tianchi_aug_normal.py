from skimage import io,data,color
from matplotlib import pyplot as plt
import numpy as np
import cv2
import os
import random
import skimage
from skimage.exposure import histogram
from skimage import feature
import json
detect_img_root='./detect_images/'
normal_img_root='./normal_images/'
aug_dir='./normal_images_aug/'

detect_name=os.listdir(detect_img_root)
normal_name=os.listdir(normal_img_root)
# # for test
# img=io.imread(img_root+file_name[0])
# # img=np.array(img)
# print(img.shape)
# cv2.imshow('image',img)
# cv2.waitKey(0)
# random.shuffle(normal_name)
# print(img.shape)
# print(img.shape)     #显示尺寸
# print(img.shape[0])  #图片高度
# print(img.shape[1])  #图片宽度
# print(img.shape[2])  #图片通道数
# print(img.size)      #显示总像素个数
# print(img.max())     #最大像素值
# print(img.min())     #最小像素值
# print(img.mean()) 

random.seed(2019)#设定随机数种子
# 正常样本扩增
result=[]
for i in range(len(normal_name)):
	# print(normal_name[i])
	img=cv2.imread(normal_img_root+normal_name[i])
	# print(img.shape)
	height,width=img.shape[0],img.shape[1]
	# print(height,width)
	defect_img=cv2.imread(detect_img_root+detect_name[i])
	# print(detect_name[i].split('.')[0].split('_')[1])
	label=detect_name[i].split('.')[0].split('_')[1]
	defect_h,defect_w=defect_img.shape[0],defect_img.shape[1]
	# print(defect_h)
	# print((height-defect_h))
	# h_range=abs(height-defect_h)
	# print((width-defect_w))
	# w_range=abs(width-defect_w)
	for j in range(5):
		xmin=random.randint(1,2446)
		ymin=random.randint(1,1000)
		xmax=xmin+defect_w
		ymax=ymin+defect_h
		print(xmin,ymin,xmax,ymax)
		print(img[xmin:xmax,ymin:ymax].shape,defect_img.shape)
		img=np.array(img)
		defect_img=np.array(defect_img)
		try:
			img[ymin:ymax,xmin:xmax]=defect_img
			result.append({'name': normal_name[i], 'category': label, 'bbox': [xmin,ymin,xmax,ymax]})
			# cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (0,255,0), 2)
			# cv2.imshow('image',img)
			# cv2.waitKey(0)
		except:
			continue
	cv2.imwrite(aug_dir+normal_name[i],img)
json_name='annotation_normal.json'
with open(json_name,'w') as fp:
        json.dump(result, fp, indent = 4, separators=(',', ': '))
	# if (defect_w*defect_h)<(32*32):






#带桌面数据处理，检测垂直边
# kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))  # 十字形结构
# for i in range(len(normal_name)):
# 	img=cv2.imread(normal_img_root+normal_name[i])
# 	img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# 	img=skimage.filters.sobel_v(img_gray)
# 	img=cv2.dilate(img, kernel)
# 	print(img.shape)
# 	row_index=[]
# 	for i in range(img.shape[1]):
# 		count_list=list(img[:,i]*255>50)
# 		row_index.append(count_list.count(True))
# 	print(np.argmax(row_index))



