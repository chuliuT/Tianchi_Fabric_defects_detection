import os
import json
import numpy as np
import pandas as pd
import cv2
import glob
import random
from PIL import Image
import time
from sklearn import metrics as mr
import shutil
random.seed(2019)
# defect_name2label = {  
#     '沾污': 1, '错花': 2, '水印': 3, '花毛': 4, '缝头': 5, '缝头印': 6, '虫粘': 7,  
#     '破洞': 8, '褶子': 9, '织疵': 10, '漏印': 11, '蜡斑': 12, '色差': 13, '网折': 14, '其他': 15  
# }

# defect_img_root='../../guangdong1_round2_train_part1_20190924/guangdong1_round2_train_part1_20190924/defect/'

# save_dir='./classification/defect'
# if not os.path.exists(save_dir):
#     os.mkdir(save_dir)
# # normal_img_root='./normal_images/'
# # aug_dir='./normal_images_aug/'
# anno_file='./anno_train_20190925.json'
# anno_result= pd.read_json(open(anno_file,"r"))
# name_list=anno_result["name"].unique()

# ring_width=10# default is 5

# result=[]
# last_result_length=0
# img_name_count=0
# for root,paths,files in os.walk(defect_img_root):
#     for path in paths:
#         img_name=path+'.jpg'
#         img_anno = anno_result[anno_result["name"] == img_name]
#         img_name_count+=1
#         # print(defect_names)
#         assert img_anno["name"].unique()[0] == img_name
#         # testimg=cv2.imread(root+path+'/'+img_name)
#         testimg=Image.open(root+path+'/'+img_name)
#         shutil.copy(root+path+'/'+img_name,save_dir+'/'+img_name)


# defect_img_root='../../guangdong1_round2_train_part2_20190924/guangdong1_round2_train_part2_20190924/defect/'

# save_dir='./classification/defect'
# if not os.path.exists(save_dir):
#     os.mkdir(save_dir)
# # normal_img_root='./normal_images/'
# # aug_dir='./normal_images_aug/'
# anno_file='./anno_train_20190925.json'
# anno_result= pd.read_json(open(anno_file,"r"))
# name_list=anno_result["name"].unique()

# ring_width=10# default is 5

# result=[]
# last_result_length=0
# img_name_count=0
# for root,paths,files in os.walk(defect_img_root):
#     for path in paths:
#         img_name=path+'.jpg'
#         img_anno = anno_result[anno_result["name"] == img_name]
#         img_name_count+=1
#         print(img_name)
#         assert img_anno["name"].unique()[0] == img_name
#         # testimg=cv2.imread(root+path+'/'+img_name)
#         testimg=Image.open(root+path+'/'+img_name)
#         shutil.copy(root+path+'/'+img_name,save_dir+'/'+img_name)

# defect_img_root='../../guangdong1_round2_train_part3_20190924/guangdong1_round2_train_part3_20190924/defect/'

# save_dir='./classification/defect'
# if not os.path.exists(save_dir):
#     os.mkdir(save_dir)
# # normal_img_root='./normal_images/'
# # aug_dir='./normal_images_aug/'
# anno_file='./anno_train_20190925.json'
# anno_result= pd.read_json(open(anno_file,"r"))
# name_list=anno_result["name"].unique()

# ring_width=10# default is 5

# result=[]
# last_result_length=0
# img_name_count=0
# for root,paths,files in os.walk(defect_img_root):
#     for path in paths:
#         img_name=path+'.jpg'
#         img_anno = anno_result[anno_result["name"] == img_name]
#         img_name_count+=1
#         print(img_name)
#         assert img_anno["name"].unique()[0] == img_name
#         # testimg=cv2.imread(root+path+'/'+img_name)
#         testimg=Image.open(root+path+'/'+img_name)
#         shutil.copy(root+path+'/'+img_name,save_dir+'/'+img_name)

# defect_img_root='../../guangdong1_round2_train2_20191004_images/defect/'

# save_dir='./classification/defect'
# if not os.path.exists(save_dir):
#     os.mkdir(save_dir)
# # normal_img_root='./normal_images/'
# # aug_dir='./normal_images_aug/'
# anno_file='./anno_train.json'
# anno_result= pd.read_json(open(anno_file,"r"))
# name_list=anno_result["name"].unique()

# ring_width=10# default is 5

# result=[]
# last_result_length=0
# img_name_count=0
# for root,paths,files in os.walk(defect_img_root):
#     for path in paths:
#         img_name=path+'.jpg'
#         img_anno = anno_result[anno_result["name"] == img_name]
#         img_name_count+=1
#         print(img_name)
#         assert img_anno["name"].unique()[0] == img_name
#         # testimg=cv2.imread(root+path+'/'+img_name)
#         testimg=Image.open(root+path+'/'+img_name)
#         shutil.copy(root+path+'/'+img_name,save_dir+'/'+img_name)

# defect_img_root='../../guangdong1_round2_train2_20191004_images/normal/'

# save_dir='./classification/normal'
# if not os.path.exists(save_dir):
#     os.mkdir(save_dir)

# for root,paths,files in os.walk(defect_img_root):
#     for path in paths:
#         img_name=path+'.jpg'
#         print(img_name)
#         shutil.copy(root+path+'/'+img_name,save_dir+'/'+img_name)


# defect_img_root='../../guangdong1_round2_train_part2_20190924/guangdong1_round2_train_part2_20190924/normal/'

# save_dir='./classification/normal'
# if not os.path.exists(save_dir):
#     os.mkdir(save_dir)


# for root,paths,files in os.walk(defect_img_root):
#     for path in paths:
#         img_name=path+'.jpg'
#         print(img_name)
#         shutil.copy(root+path+'/'+img_name,save_dir+'/'+img_name)

# defect_img_root='../../guangdong1_round2_train2_20191004_images/defect/'
# count=0

# save_dir='./classification/normal'
# if not os.path.exists(save_dir):
#     os.mkdir(save_dir)
# # normal_img_root='./normal_images/'
# # aug_dir='./normal_images_aug/'
# # anno_file='./anno_train.json'
# # anno_result= pd.read_json(open(anno_file,"r"))
# # name_list=anno_result["name"].unique()

# # ring_width=10# default is 5
# for root,paths,files in os.walk(defect_img_root):
#     for path in paths:
#         # img_name=path+'.jpg'
#         # img_anno = anno_result[anno_result["name"] == img_name]
#         # img_name_count+=1
#         count+=1
#         # assert img_anno["name"].unique()[0] == img_name
#         # testimg=cv2.imread(root+path+'/'+img_name)
#         # testimg=Image.open(root+path+'/'+img_name)
#         template_img_name='template_'+path.split('_')[0]+'.jpg'
#         print(template_img_name)
#         shutil.copy(root+path+'/'+template_img_name,save_dir+'/'+str(count)+'_'+template_img_name)

# defect_img_root='../../guangdong1_round2_train_part1_20190924/guangdong1_round2_train_part1_20190924/defect/'
# # count=0

# # save_dir='./classification/normal'
# # if not os.path.exists(save_dir):
# #     os.mkdir(save_dir)
# # normal_img_root='./normal_images/'
# # aug_dir='./normal_images_aug/'
# # anno_file='./anno_train.json'
# # anno_result= pd.read_json(open(anno_file,"r"))
# # name_list=anno_result["name"].unique()

# # ring_width=10# default is 5
# for root,paths,files in os.walk(defect_img_root):
#     for path in paths:
#         # img_name=path+'.jpg'
#         # img_anno = anno_result[anno_result["name"] == img_name]
#         # img_name_count+=1
#         count+=1
#         # assert img_anno["name"].unique()[0] == img_name
#         # testimg=cv2.imread(root+path+'/'+img_name)
#         # testimg=Image.open(root+path+'/'+img_name)
#         template_img_name='template_'+path.split('_')[0]+'.jpg'
#         print(template_img_name)
#         shutil.copy(root+path+'/'+template_img_name,save_dir+'/'+str(count)+'_'+template_img_name)

# defect_img_root='../../guangdong1_round2_train_part2_20190924/guangdong1_round2_train_part2_20190924/defect/'
# # count=0

# # save_dir='./classification/normal'
# # if not os.path.exists(save_dir):
# #     os.mkdir(save_dir)
# # normal_img_root='./normal_images/'
# # aug_dir='./normal_images_aug/'
# # anno_file='./anno_train.json'
# # anno_result= pd.read_json(open(anno_file,"r"))
# # name_list=anno_result["name"].unique()

# # ring_width=10# default is 5
# for root,paths,files in os.walk(defect_img_root):
#     for path in paths:
#         # img_name=path+'.jpg'
#         # img_anno = anno_result[anno_result["name"] == img_name]
#         # img_name_count+=1
#         count+=1
#         # assert img_anno["name"].unique()[0] == img_name
#         # testimg=cv2.imread(root+path+'/'+img_name)
#         # testimg=Image.open(root+path+'/'+img_name)
#         template_img_name='template_'+path.split('_')[0]+'.jpg'
#         print(template_img_name)
#         shutil.copy(root+path+'/'+template_img_name,save_dir+'/'+str(count)+'_'+template_img_name)

# defect_img_root='../../guangdong1_round2_train_part3_20190924/guangdong1_round2_train_part3_20190924/defect/'
# # count=0

# # save_dir='./classification/normal'
# # if not os.path.exists(save_dir):
# #     os.mkdir(save_dir)
# # normal_img_root='./normal_images/'
# # aug_dir='./normal_images_aug/'
# # anno_file='./anno_train.json'
# # anno_result= pd.read_json(open(anno_file,"r"))
# # name_list=anno_result["name"].unique()

# # ring_width=10# default is 5
# for root,paths,files in os.walk(defect_img_root):
#     for path in paths:
#         # img_name=path+'.jpg'
#         # img_anno = anno_result[anno_result["name"] == img_name]
#         # img_name_count+=1
#         count+=1
#         # assert img_anno["name"].unique()[0] == img_name
#         # testimg=cv2.imread(root+path+'/'+img_name)
#         # testimg=Image.open(root+path+'/'+img_name)
#         template_img_name='template_'+path.split('_')[0]+'.jpg'
#         print(template_img_name)
#         shutil.copy(root+path+'/'+template_img_name,save_dir+'/'+str(count)+'_'+template_img_name)


images_defect_list=os.listdir('./classification/defect')
print(len(images_defect_list))
images_normal_list=os.listdir('./classification/normal')
random.shuffle(images_defect_list)

keep_normal_list=random.sample(images_normal_list,len(images_defect_list))
print(len(keep_normal_list))
for idx,img_name in enumerate(images_normal_list):
    if img_name not in keep_normal_list:
        os.remove('./classification/normal/'+img_name)


images_defect_list=os.listdir('./classification/defect')
print(len(images_defect_list))
images_normal_list=os.listdir('./classification/normal')
print(len(images_normal_list))