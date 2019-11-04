import os
import json
import numpy as np
import pandas as pd
import cv2
import glob
import random
from PIL import Image
import time

random.seed(2019)
defect_name2label = {  
    '沾污': 1, '错花': 2, '水印': 3, '花毛': 4, '缝头': 5, '缝头印': 6, '虫粘': 7,  
    '破洞': 8, '褶子': 9, '织疵': 10, '漏印': 11, '蜡斑': 12, '色差': 13, '网折': 14, '其他': 15  
}

defect_img_root='./normal_aug/'


anno_file='./Duck_inject_normal.json'
anno_result= pd.read_json(open(anno_file,"r"))
name_list=anno_result["name"].unique()
# for img_name in name_list:
ring_width=5

result=[]


for img_name in os.listdir(defect_img_root):
        img_anno = anno_result[anno_result["name"] == img_name]
        bboxs = img_anno["bbox"].tolist()
        # print(bboxs)
        defect_names = img_anno["defect_name"].tolist()
        defect_names = [defect_name2label[x] for x in defect_names]
        print(defect_names)
        # assert img_anno["name"].unique()[0] == img_name
        testimg=cv2.imread(defect_img_root+'/'+img_name)
        
        for idx in range(len(bboxs)):
            pts=bboxs[idx]
            d_name=defect_names[idx]
            xmin=pts[0]
            ymin=pts[1]
            xmax=pts[2]
            ymax=pts[3]
            defect_h=abs(xmax-xmin)
            defect_w=abs(ymax-ymin)
            w_h=round(defect_w/defect_h,2)
            h_w=round(defect_h/defect_w,2)
            cv2.putText(testimg, str(d_name),(int(xmin),int(ymin)),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
            print('w_h',w_h)
            print('h_w',h_w)
            print('defect_size:',(ymax-ymin)*(xmax-xmin))
            # cv2.putText(testimg, str(w_h),(int(xmin+10),int(ymin)),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
            # cv2.putText(testimg, str(h_w),(int(xmin+30),int(ymin)),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
            cv2.rectangle(testimg, (int(xmin),int(ymin)), (int(xmax),int(ymax)), (0,0,255), 2)
        
        cv2.namedWindow("testimg",0);
        cv2.resizeWindow("testimg", 1200, 800);
        cv2.imshow('testimg',testimg)
        cv2.waitKey(0)
 