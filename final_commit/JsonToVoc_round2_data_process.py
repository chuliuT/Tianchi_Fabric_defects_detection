
import os
import numpy as np
import codecs
import json
from glob import glob
import cv2
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd

defect_name2label = {  
    '沾污': 1, '错花': 2, '水印': 3, '花毛': 4, '缝头': 5, '缝头印': 6, '虫粘': 7,  
    '破洞': 8, '褶子': 9, '织疵': 10, '漏印': 11, '蜡斑': 12, '色差': 13, '网折': 14, '其他': 15  
}

#1.标签路径
# image_path = "./guangdong1_round1_train1_20190818/defect_Images/"              #原始labelme标注数据路径
saved_path = "./VOC2020/"                #保存路径

#2.创建要求文件夹
if not os.path.exists(saved_path + "Annotations"):
    os.makedirs(saved_path + "Annotations")
if not os.path.exists(saved_path + "JPEGImages/"):
    os.makedirs(saved_path + "JPEGImages/")
if not os.path.exists(saved_path + "ImageSets/Main/"):
    os.makedirs(saved_path + "ImageSets/Main/")
#cp images
#part1
image_path = "./guangdong1_round2_train_part1_20190924/guangdong1_round2_train_part1_20190924/defect/"  
image_fold=os.listdir(image_path)
cp_path = saved_path+"/JPEGImages"

for i in range(1,len(image_fold)):
    print(i)
    image_files = glob(image_path+image_fold[i]+'/'+ "*.jpg")
    shutil.copy(image_files[0],cp_path)
print('copy part1 defect images done......')
#part 2
image_path = "./guangdong1_round2_train_part2_20190924/guangdong1_round2_train_part2_20190924/defect/"  
image_fold=os.listdir(image_path)
cp_path = saved_path+"/JPEGImages"

for i in range(1,len(image_fold)):
    print(i)
    image_files = glob(image_path+image_fold[i]+'/'+ "*.jpg")
    shutil.copy(image_files[0],cp_path)
print('copy part2 defect images done......')
#part3
image_path = "./guangdong1_round2_train_part3_20190924/guangdong1_round2_train_part3_20190924/defect/"  
image_fold=os.listdir(image_path)
cp_path = saved_path+"/JPEGImages"

for i in range(1,len(image_fold)):
    print(i)
    image_files = glob(image_path+image_fold[i]+'/'+ "*.jpg")
    shutil.copy(image_files[0],cp_path)
###
print('copy three part defect images done......')

json_file  = "./anno_train_20190925.json"
files = [1]
#4.读取标注信息并写入 xml
for tmp in files:
    json_filename = json_file
    anno_result = pd.read_json(open(json_filename,"r",encoding="utf-8"))
    
    name_list=anno_result["name"].unique()
    for img_name in name_list:
            img_anno = anno_result[anno_result["name"] == img_name]
            bboxs = img_anno["bbox"].tolist()
            defect_names = img_anno["defect_name"].tolist()
            assert img_anno["name"].unique()[0] == img_name
            print(img_name,bboxs,defect_names)
            with codecs.open(saved_path + "Annotations/" + img_name.split('.')[0] + ".xml","w","utf-8") as xml:
                height, width, channels = 1696,4096,3

                xml.write('<annotation>\n')
                xml.write('\t<folder>' + 'UAV_data' + '</folder>\n')
                xml.write('\t<filename>' + img_name + '</filename>\n')
                xml.write('\t<size>\n')
                xml.write('\t\t<width>'+ str(width) + '</width>\n')
                xml.write('\t\t<height>'+ str(height) + '</height>\n')
                xml.write('\t\t<depth>' + str(channels) + '</depth>\n')
                xml.write('\t</size>\n')
                xml.write('\t\t<segmented>0</segmented>\n')
                
                
                for bbox, defect_name in zip(bboxs, defect_names):
                    label= defect_name2label[defect_name]
                    points = np.array(bbox)
                    xmin = points[0]
                    xmax = points[2]
                    ymin = points[1]
                    ymax = points[3]
                    if xmax <= xmin:
                        pass
                    elif ymax <= ymin:
                        pass
                    else:
                        xml.write('\t<object>\n')
                        xml.write('\t\t<name>'+str(label)+'</name>\n')
                        xml.write('\t\t<pose>Unspecified</pose>\n')
                        xml.write('\t\t<truncated>1</truncated>\n')
                        xml.write('\t\t<difficult>0</difficult>\n')
                        xml.write('\t\t<bndbox>\n')
                        xml.write('\t\t\t<xmin>' + str(xmin) + '</xmin>\n')
                        xml.write('\t\t\t<ymin>' + str(ymin) + '</ymin>\n')
                        xml.write('\t\t\t<xmax>' + str(xmax) + '</xmax>\n')
                        xml.write('\t\t\t<ymax>' + str(ymax) + '</ymax>\n')
                        xml.write('\t\t</bndbox>\n')
                        xml.write('\t</object>\n')
                    # print(multi['name'],xmin,ymin,xmax,ymax,label)
                xml.write('</annotation>')
        
# # 5.复制图片到 VOC2007/JPEGImages/下
# image_files = glob(image_path + "*.jpg")
# print("copy image files to VOC20/JPEGImages/")
# for image in tqdm(image_files):
#     shutil.copy(image,saved_path +"JPEGImages/")

#6.split files for txt
txtsavepath = saved_path + "ImageSets/Main/"
ftrainval = open(txtsavepath+'/trainval.txt', 'w')
ftest = open(txtsavepath+'/test.txt', 'w')
ftrain = open(txtsavepath+'/train.txt', 'w')
fval = open(txtsavepath+'/val.txt', 'w')
total_files = os.listdir("./VOC2020/Annotations/")

total_files = [i.split("/")[-1].split(".xml")[0] for i in total_files]
#test_filepath = ""
for file in total_files:
    ftrainval.write(file + "\n")
    
#split
train_files,val_files = train_test_split(total_files,test_size=0.15,random_state=42)
#train
for file in train_files:
    ftrain.write(file + "\n")
#val
for file in val_files:
    fval.write(file + "\n")

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()   