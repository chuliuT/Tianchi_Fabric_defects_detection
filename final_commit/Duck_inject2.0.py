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

random.seed(2019)
# defect_name2label = {  
#     '沾污': 1, '错花': 2, '水印': 3, '花毛': 4, '缝头': 5, '缝头印': 6, '虫粘': 7,  
#     '破洞': 8, '褶子': 9, '织疵': 10, '漏印': 11, '蜡斑': 12, '色差': 13, '网折': 14, '其他': 15  
# }
aug_name=['沾污', '错花', '水印', '花毛']

defect_img_root='../../guangdong1_round2_train_part1_20190924/guangdong1_round2_train_part1_20190924/defect/'

save_dir='./normal_aug/'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
# normal_img_root='./normal_images/'
# aug_dir='./normal_images_aug/'
anno_file='./anno_train_20190925.json'
anno_result= pd.read_json(open(anno_file,"r"))
name_list=anno_result["name"].unique()

ring_width=10# default is 5

result=[]
last_result_length=0
img_name_count=0
for root,paths,files in os.walk(defect_img_root):
    for path in paths:
        img_name=path+'.jpg'
        img_anno = anno_result[anno_result["name"] == img_name]
        bboxs = img_anno["bbox"].tolist()
        # print(bboxs)
        img_name_count+=1
        defect_names = img_anno["defect_name"].tolist()
        # defect_names = [defect_name2label[x] for x in defect_names]
        print(defect_names)
        assert img_anno["name"].unique()[0] == img_name
        # testimg=cv2.imread(root+path+'/'+img_name)
        testimg=Image.open(root+path+'/'+img_name)
        template_img_name='template_'+path.split('_')[0]+'.jpg'
        # temp_img=cv2.imread(root+path+'/'+template_img_name)
        temp_img=Image.open(root+path+'/'+template_img_name)
        save_temp_name='template_'+path.split('_')[0]+str(img_name_count)+'.jpg'
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
            # cv2.putText(testimg, str(d_name),(int(xmin),int(ymin)),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
            # 用于长条判断
            # print('w_h',w_h)
            # print('h_w',h_w)
            # print('defect_size:',(ymax-ymin)*(xmax-xmin))
            # cv2.putText(testimg, str(w_h),(int(xmin+10),int(ymin)),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
            # cv2.putText(testimg, str(h_w),(int(xmin+30),int(ymin)),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
            # cv2.rectangle(testimg, (int(xmin),int(ymin)), (int(xmax),int(ymax)), (0,0,255), 2)
            # 筛选长条的采点位置
            if w_h > 5 or h_w >5 or (ymax-ymin)*(xmax-xmin)>300000:#TODO: 这里的两个5感觉有问题，300000的应该丢弃
                left_top_x=random.randint(1,3)
                left_top_y=random.randint(1,testimg.size[1])
            else:
                left_top_x=random.randint(1,testimg.size[0])
                left_top_y=random.randint(1,testimg.size[1])
            # print(left_top_x,left_top_y)
            mask=np.zeros_like(temp_img)
            if d_name in aug_name:
                scale=random.randint(3,5)
                mask[int(scale*(left_top_y-ring_width)):int(scale*(left_top_y+defect_w+ring_width)),int(scale*(left_top_x-ring_width)):int(scale*(left_top_x+defect_h+ring_width))]=255
                mask[int(left_top_y):int(scale*(left_top_y+defect_w)),int(left_top_x):int(scale*(left_top_x+defect_h))]=0

                # cv2.namedWindow("mask",0);
                # cv2.resizeWindow("mask", 1200, 800);
                # cv2.imshow('mask',mask)
                # cv2.imwrite('mask.jpg',mask)
                # cv2.waitKey(0)
                patch=testimg.crop((xmin,ymin,xmax,ymax))
                #====相似度计算==============================================================================================#
                patch1=patch.copy()
                patch2=temp_img.crop((left_top_x,left_top_y,int(left_top_x+patch1.size[0]),int(left_top_y+patch1.size[1])))

                # print('bbox:',(left_top_x,left_top_y,int(left_top_x+(xmax-xmin)),int(left_top_y+(ymax-ymin))))
                # print(patch1.size[0],patch1.size[1])
                # print(patch1.size,patch2.size)
                patch2.resize((patch1.size[0],patch1.size[1]))
                patch1=np.resize(patch1,-1)
                patch2=np.resize(patch2,-1)
                # print(patch1.shape)
                # print(patch2.shape)
                mutual_infor=mr.mutual_info_score(patch1,patch2)
                print(mutual_infor)
                #==================================================================================================#
                if mutual_infor>0.8:
                    print(patch.size)
                    patch=patch.resize((patch.size[0]*scale,patch.size[1]*scale))
                    print(patch.size)
                    temp_img.paste(patch,(left_top_x,left_top_y))
                    temp_img = cv2.cvtColor(np.asarray(temp_img),cv2.COLOR_RGB2BGR)
                    temp_img = cv2.inpaint(temp_img,mask[:,:,0],3,cv2.INPAINT_TELEA)
                    temp_img = Image.fromarray(cv2.cvtColor(temp_img,cv2.COLOR_BGR2RGB))
                    result.append({'name': save_temp_name, 'defect_name': d_name, 'bbox': [left_top_x,left_top_y,left_top_x+defect_h*scale,left_top_y+defect_w*scale]})

                else:
                    continue
            else:  
                mask[int(left_top_y-ring_width):int(left_top_y+defect_w+ring_width),int(left_top_x-ring_width):int(left_top_x+defect_h+ring_width)]=255
                mask[int(left_top_y):int(left_top_y+defect_w),int(left_top_x):int(left_top_x+defect_h)]=0
                
                # cv2.namedWindow("mask",0);
                # cv2.resizeWindow("mask", 1200, 800);
                # cv2.imshow('mask',mask)
                # cv2.imwrite('mask.jpg',mask)
                # cv2.waitKey(0)
                patch=testimg.crop((xmin,ymin,xmax,ymax))
                #====相似度计算==============================================================================================#
                patch1=patch.copy()
                patch2=temp_img.crop((left_top_x,left_top_y,int(left_top_x+patch1.size[0]),int(left_top_y+patch1.size[1])))

                # print('bbox:',(left_top_x,left_top_y,int(left_top_x+(xmax-xmin)),int(left_top_y+(ymax-ymin))))
                # print(patch1.size[0],patch1.size[1])
                # print(patch1.size,patch2.size)
                patch2.resize((patch1.size[0],patch1.size[1]))
                patch1=np.resize(patch1,-1)
                patch2=np.resize(patch2,-1)
                # print(patch1.shape)
                # print(patch2.shape)
                mutual_infor=mr.mutual_info_score(patch1,patch2)
                print(mutual_infor)
                #==================================================================================================#
                if mutual_infor>0.8:
                    temp_img.paste(patch,(left_top_x,left_top_y))
                    temp_img = cv2.cvtColor(np.asarray(temp_img),cv2.COLOR_RGB2BGR)
                    temp_img = cv2.inpaint(temp_img,mask[:,:,0],3,cv2.INPAINT_TELEA)
                    temp_img = Image.fromarray(cv2.cvtColor(temp_img,cv2.COLOR_BGR2RGB))
                    result.append({'name': save_temp_name, 'defect_name': d_name, 'bbox': [left_top_x,left_top_y,left_top_x+defect_h,left_top_y+defect_w]})

                else:
                    continue
     
            # cv2.rectangle(temp_img, (int(left_top_x),int(left_top_y)), (int(left_top_x+defect_h),int(left_top_y+defect_w)), (0,0,255), 2)
        # TODO 这里会引入正常图
        temp_img.save(save_dir+save_temp_name)

        #test path
        json_name='./Duck_inject_normal.json'
        with open(json_name,'w') as fp:
                json.dump(result, fp, indent = 4, separators=(',', ': ')) 




# json_name='./Duck_inject_normal.json'
# with open(json_name,'w') as fp:
#         json.dump(result, fp, indent = 4, separators=(',', ': '))            
        # testimg.show()
        # temp_img.show()
        # # sys.pause(0)
        # time.sleep(2)
        # print(defect_img_root+defect_name[1]+'/'+defect_name[1]+'.jpg')
        # cv2.namedWindow("testimg",0);
        # cv2.resizeWindow("testimg", 1200, 800);
        # cv2.imshow('testimg',testimg)

        # cv2.namedWindow("temp_img",0);
        # cv2.resizeWindow("temp_img", 1200, 800);
        # cv2.imshow('temp_img',temp_img)
        # cv2.waitKey(0)
# print(defect_name)
# testimg=cv2.imread(defect_img_root+defect_name[1]+'/'+defect_name[1]+'.jpg')
# print(defect_img_root+defect_name[1]+'/'+defect_name[1]+'.jpg')
# cv2.imshow('testimg',testimg)
# cv2.waitKey(0)
