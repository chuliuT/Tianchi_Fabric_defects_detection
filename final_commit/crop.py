# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 09:25:17 2018
裁剪
@author: zxl
"""

import cv2
import os
import random
import voc_xml
import utils
from voc_xml import CreateXML
import xml.etree.cElementTree as ET

def readAnnotations(xml_path):
    et = ET.parse(xml_path)
    element = et.getroot()
    element_objs = element.findall('object')
 
    results = []
    for element_obj in element_objs:
        result = []
        class_name = element_obj.find('name').text
 
        obj_bbox = element_obj.find('bndbox')
        x1 = int(round(float(obj_bbox.find('xmin').text)))
        y1 = int(round(float(obj_bbox.find('ymin').text)))
        x2 = int(round(float(obj_bbox.find('xmax').text)))
        y2 = int(round(float(obj_bbox.find('ymax').text)))
 
        result.append(int(x1))
        result.append(int(y1))
        result.append(int(x2))
        result.append(int(y2))
 
        results.append(result)
    return results
def crop_img(src,top_left_x,top_left_y,crop_w,crop_h):
    '''裁剪图像
    Args:
        src: 源图像
        top_left,top_right:裁剪图像左上角坐标
        crop_w,crop_h：裁剪图像宽高
    return：
        crop_img:裁剪后的图像
        None:裁剪尺寸错误
    '''
    rows,cols,n_channel = src.shape
    row_min,col_min = int(top_left_y), int(top_left_x)
    row_max,col_max = int(row_min + crop_h), int(col_min + crop_w)
    if row_max > rows or col_max > cols:
        print("crop size err: src->%dx%d,crop->top_left(%d,%d) %dx%d"%(cols,rows,col_min,row_min,int(crop_w),int(crop_h)))
        return None
    crop_img = src[row_min:row_max,col_min:col_max]
    return crop_img

def crop_xy(x,y,top_left_x,top_left_y,crop_w,crop_h):
    ''' 坐标平移变换
    Args:
        x,y:待变换坐标
        top_left_x,top_left_y:裁剪图像左上角坐标
        crop_w,crop_h:裁剪部分图像宽高
    return:
        crop_x,crop_y
    '''
    crop_x = int(x - top_left_x)
    crop_y = int(y - top_left_y)
    crop_x = utils.confine(crop_x,0,crop_w-1)
    crop_y = utils.confine(crop_y,0,crop_h-1)
    return crop_x,crop_y
    

def crop_box(box,top_left_x,top_left_y,crop_w,crop_h,iou_thr=0.5):
    '''目标框坐标平移变换
    Args:
        box:目标框坐标[xmin,ymin,xmax,ymax]
        top_left_x,top_left_y:裁剪图像左上角坐标
        crop_w,crop_h:裁剪部分图像宽高
        iou_thr: iou阈值,去除裁剪后过小目标
    return:
        crop_box:平移变换结果[xmin,ymin,xmax,ymax]
    '''
    xmin,ymin = crop_xy(box[0],box[1],top_left_x,top_left_y,crop_w,crop_h)
    xmax,ymax = crop_xy(box[2],box[3],top_left_x,top_left_y,crop_w,crop_h)
    croped_box = [xmin,ymin,xmax,ymax]
    if utils.calc_iou([0,0,box[2]-box[0],box[3]-box[1]],[0,0,xmax-xmin,ymax-ymin]) < iou_thr:
        croped_box = [0,0,0,0]
    return croped_box


def crop_xml(crop_img_name,xml_tree,top_left_x,top_left_y,crop_w,crop_h,iou_thr=0.5):
    '''xml目标框裁剪变换
    Args:
        crop_img_name:裁剪图片命名
        xml_tree：待crop的xml ET.parse()
        top_left_x,top_left_y: 裁剪图像左上角坐标
        crop_w,crop_h: 裁剪图像宽高
        iou_thr: iou阈值
    return:
        createdxml : 创建的xml CreateXML对象         
    '''
    root = xml_tree.getroot()
    size = root.find('size')
    depth = int(size.find('depth').text)
    createdxml = CreateXML(crop_img_name,int(crop_w),int(crop_h),depth)
    for obj in root.iter('object'):
        obj_name = obj.find('name').text
        xml_box = obj.find('bndbox')
        xmin = float(xml_box.find('xmin').text)
        ymin = float(xml_box.find('ymin').text)
        xmax = float(xml_box.find('xmax').text)
        ymax = float(xml_box.find('ymax').text)
        box = crop_box([xmin,ymin,xmax,ymax],top_left_x,top_left_y,crop_w,crop_h,iou_thr)       
        if (box[0] >= box[2]) or (box[1] >= box[3]):
            continue
        createdxml.add_object_node(obj_name,box[0],box[1],box[2],box[3])
    return createdxml

def crop_img_xml(img,xml_tree,crop_img_name,top_left_x,top_left_y,crop_w,crop_h,iou_thr):
    '''裁剪图像和xml目标框
    Args:
        img：源图像
        crop_img_name:裁剪图片命名
        xml_tree：待crop的xml ET.parse()
        top_left_x,top_left_y: 裁剪图像左上角坐标
        crop_w,crop_h: 裁剪图像宽高
        iou_thr: iou阈值
    return:
        croped_img,croped_xml : 裁剪完成的图像和xml文件
        None:裁剪尺寸错误
    '''   
    croped_img = crop_img(img,top_left_x,top_left_y,crop_w,crop_h)
    if croped_img is None:
        return None
    croped_xml = crop_xml(crop_img_name,xml_tree,top_left_x,top_left_y,crop_w,crop_h,iou_thr)
    return croped_img,croped_xml



def crop_img_xml_from_dir(imgs_dir,xmls_dir,imgs_save_dir,xmls_save_dir,img_suffix,name_suffix,\
                                 crop_type='RANDOM_CROP',crop_n=1,dsize=(0,0),fw=1.0,fh=1.0,random_wh=False,iou_thr=0.5):
    '''随机裁剪指定路径下的图片和xml
    Args:
        imgs_dir,xmls_dir: 待放缩图片、原始xml文件存储路径
        imgs_save_dir，xmls_save_dir: 处理完成的图片、xml文件存储路径
        img_suffix: 图片可能的后缀名['.jpg','.png','.bmp',..]
        name_suffix: 处理完成的图片、xml的命名标识
        crop_type:裁剪风格 ['RANDOM_CROP','CENTER_CROP','FIVE_CROP']
        crop_n: 每原图生成裁剪图个数
        dsize:指定crop宽高（w,h），与random_wh==True互斥生效
        fw,fh: 当random_wh==False时为crop比例，否则为随机crop的宽高比例下限
        random_wh：随机选定裁剪宽高
        iou_thr: iou阈值
    '''       
    for root,dirs,files in os.walk(xmls_dir):
        for xml_name in files:
            xml_file = os.path.join(xmls_dir,xml_name)
            #print(xml_file)
            img_file = None
            for suffix in img_suffix:
                #print(os.path.join(imgs_dir,xml_name.split('.')[0]+suffix))
                if os.path.exists(os.path.join(imgs_dir,xml_name.split('.')[0]+suffix)):
                    img_file = os.path.join(imgs_dir,xml_name.split('.')[0]+suffix)
                    break
            if img_file is None:
                print("there has no image for ",xml_name)
                continue
            img = cv2.imread(img_file)
            imgh,imgw,n_channels = img.shape
            bboxes = readAnnotations(xml_file)
            # print("img: {},  box: {}".format(image_path, bboxes))
            x1,y1,x2,y2=[],[],[],[]
            for box in bboxes:
                x1.append(box[0])
                y1.append(box[1])
                x2.append(box[2])
                y2.append(box[3])
            x1.sort()
            y1.sort()
            x2.sort()
            y2.sort()
            # print(annotation_path,x1[0],y1[0],x2[-1],y2[-1])
            
            if crop_type == 'CENTER_CROP':
                crop_n = 1
            elif crop_type == 'FIVE_CROP':
                crop_n = 5            
            elif crop_type == 'TwentyFour_CROP':
                crop_n = 21  
            for i in range(crop_n):
                crop_imgw,crop_imgh = dsize
                if dsize == (0,0) and not random_wh:
                    crop_imgw = int(imgw*fw)
                    crop_imgh = int(imgh*fh)
                elif random_wh:
                    crop_imgw = int(imgw*(fw + random.random()*(1-fw)))
                    crop_imgh = int(imgh*(fh + random.random()*(1-fh)))
               
                if crop_type == 'RANDOM_CROP':
                    crop_top_left_x,crop_top_left_y =  random.randint(x1[0],x2[-1]-crop_imgw-1),random.randint(y1[0],y2[-1]-crop_imgh-1)
                elif crop_type == 'CENTER_CROP':
                    crop_top_left_x,crop_top_left_y = int(imgw/2-crop_imgw/2),int(imgh/2-crop_imgh/2)
                elif crop_type == 'FIVE_CROP':
                    if i == 0:
                        crop_top_left_x,crop_top_left_y = 0,0
                    elif i == 1:
                        crop_top_left_x,crop_top_left_y = imgw-crop_imgw-1,0
                    elif i == 2:
                        crop_top_left_x,crop_top_left_y = 0,imgh-crop_imgh-1
                    elif i == 3:
                        crop_top_left_x,crop_top_left_y = imgw-crop_imgw-1,imgh-crop_imgh-1
                    else:
                        crop_top_left_x,crop_top_left_y = int(imgw/2-crop_imgw/2),int(imgh/2-crop_imgh/2)
                elif crop_type == 'TwentyFour_CROP':
                    if i == 0:
                        crop_top_left_x,crop_top_left_y = 0,0
                    elif i == 1:
                        crop_top_left_x,crop_top_left_y = 0,512
                    elif i == 2:
                        crop_top_left_x,crop_top_left_y = 0,1024
                    elif i == 3:
                        crop_top_left_x,crop_top_left_y = 512,0
                    elif i == 4:
                        crop_top_left_x,crop_top_left_y = 512,512
                    elif i == 5:
                        crop_top_left_x,crop_top_left_y = 512,1024
                    elif i == 6:
                        crop_top_left_x,crop_top_left_y = 1024,0
                    elif i == 7:
                        crop_top_left_x,crop_top_left_y = 1024,512
                    elif i == 8:
                        crop_top_left_x,crop_top_left_y = 1024,1024
                    elif i == 9:
                        crop_top_left_x,crop_top_left_y = 1536,0
                    elif i == 10:
                        crop_top_left_x,crop_top_left_y = 1536,512
                    elif i == 11:
                        crop_top_left_x,crop_top_left_y = 1536,1024
                    elif i == 12:
                        crop_top_left_x,crop_top_left_y = 2560,0
                    elif i == 13:
                        crop_top_left_x,crop_top_left_y = 2560,512
                    elif i == 14:
                        crop_top_left_x,crop_top_left_y = 2560,1024
                    elif i == 15:
                        crop_top_left_x,crop_top_left_y = 3072,0
                    elif i == 16:
                        crop_top_left_x,crop_top_left_y = 3072,512
                    elif i == 17:
                        crop_top_left_x,crop_top_left_y = 3072,1024
                    elif i == 18:
                        crop_top_left_x,crop_top_left_y = 3584,0
                    elif i == 19:
                        crop_top_left_x,crop_top_left_y = 3584,512
                    else:
                        crop_top_left_x,crop_top_left_y = 3584,1024
                    # elif i == 21:
                    #     crop_top_left_x,crop_top_left_y = 3764,0
                    # elif i == 22:
                    #     crop_top_left_x,crop_top_left_y = 3764,512
                    # else:
                    #     crop_top_left_x,crop_top_left_y = 3764,1024
                    # else:
                    #     crop_top_left_x,crop_top_left_y = int(imgw/2-crop_imgw/2),int(imgh/2-crop_imgh/2)
                else:
                    print('crop type wrong! expect [RANDOM_CROP,CENTER_CROP,FIVE_CROP]' )
                 
                croped_img_name = xml_name.split('.')[0]+'_'+name_suffix +\
                            str(crop_top_left_x)+'_'+str(crop_top_left_y)+\
                            '_wh'+str(crop_imgw)+'x'+str(crop_imgh)+\
                            '.'+img_file.split('.')[-1]
                croped = crop_img_xml(img,voc_xml.get_xml_tree(xml_file),croped_img_name,crop_top_left_x,crop_top_left_y,crop_imgw,crop_imgh,iou_thr)
                imgcrop,xmlcrop = croped[0],croped[1]
                cv2.imwrite(os.path.join(imgs_save_dir,croped_img_name),imgcrop)
                xmlcrop.save_xml(xmls_save_dir,croped_img_name.split('.')[0]+'.xml')

def readAnnotations(xml_path):
    et = ET.parse(xml_path)
    element = et.getroot()
    element_objs = element.findall('object')
 
    results = []
    for element_obj in element_objs:
        result = []
        class_name = element_obj.find('name').text
 
        obj_bbox = element_obj.find('bndbox')
        x1 = int(round(float(obj_bbox.find('xmin').text)))
        y1 = int(round(float(obj_bbox.find('ymin').text)))
        x2 = int(round(float(obj_bbox.find('xmax').text)))
        y2 = int(round(float(obj_bbox.find('ymax').text)))
 
        result.append(int(x1))
        result.append(int(y1))
        result.append(int(x2))
        result.append(int(y2))
        # result.append(222)
 
        results.append(result)
    return results

def crop_imgs_without_label(imgs_dir,imgs_save_dir,name_suffix,crop_type='RANDOM_CROP',\
                            crop_n=1,dsize=(0,0),fw=1.0,fh=1.0,random_wh=False):
    '''仅裁剪图片，不带标签
    Args：
        imgs_dir: 待放缩图片、原始xml文件存储路径
        imgs_save_dir: 处理完成的图片、xml文件存储路径
        name_suffix: 处理完成的图片、xml的命名标识
        crop_type:裁剪风格 ['RANDOM_CROP','CENTER_CROP','FIVE_CROP']
        crop_n: 每原图生成裁剪图个数
        dsize:指定crop宽高（w,h），与random_wh==True互斥生效
        fw,fh: 当random_wh==False时为crop比例，否则为随机crop的宽高比例下限
        random_wh：随机选定裁剪宽高  
    '''
    imgcount = utils.fileCountIn(imgs_dir)
    count = 0
    for root,dirs,files in os.walk(imgs_dir):
        for file in files:
            img_file = os.path.join(imgs_dir,file)
            img = cv2.imread(img_file)
            imgh,imgw,n_channels = img.shape
            
            
            if crop_type == 'CENTER_CROP':
                crop_n = 1
            elif crop_type == 'FIVE_CROP':
                crop_n = 5            
            
            for i in range(crop_n):
                crop_imgw,crop_imgh = dsize
                if dsize == (0,0) and not random_wh:
                    crop_imgw = int(imgw*fw)
                    crop_imgh = int(imgh*fh)
                elif random_wh:
                    crop_imgw = int(imgw*(fw + random.random()*(1-fw)))
                    crop_imgh = int(imgh*(fh + random.random()*(1-fh)))
               
                if crop_type == 'RANDOM_CROP':
                    crop_top_left_x,crop_top_left_y =  random.randint(0,imgw-crop_imgw-1),random.randint(0,imgh-crop_imgh-1)
                elif crop_type == 'CENTER_CROP':
                    crop_top_left_x,crop_top_left_y = int(imgw/2-crop_imgw/2),int(imgh/2-crop_imgh/2)
                elif crop_type == 'FIVE_CROP':
                    if i == 0:
                        crop_top_left_x,crop_top_left_y = 0,0
                    elif i == 1:
                        crop_top_left_x,crop_top_left_y = imgw-crop_imgw-1,0
                    elif i == 2:
                        crop_top_left_x,crop_top_left_y = 0,imgh-crop_imgh-1
                    elif i == 3:
                        crop_top_left_x,crop_top_left_y = imgw-crop_imgw-1,imgh-crop_imgh-1
                    else:
                        crop_top_left_x,crop_top_left_y = int(imgw/2-crop_imgw/2),int(imgh/2-crop_imgh/2)

                else:
                    print('crop type wrong! expect [RANDOM_CROP,CENTER_CROP,FIVE_CROP]' )  
                    
                croped_img_name = file.split('.')[0]+'_'+name_suffix +\
                            str(crop_top_left_x)+'_'+str(crop_top_left_y)+\
                            '_wh'+str(crop_imgw)+'x'+str(crop_imgh)+\
                            '.jpg'
                croped_img = crop_img(img,crop_top_left_x,crop_top_left_y,crop_imgw,crop_imgh)
                cv2.imwrite(os.path.join(imgs_save_dir,croped_img_name),croped_img)
            count += 1
            if count % 10 == 0:
                print('[%d|%d] %d%%'%(count,imgcount,count*100/imgcount))
            
                

def main():
    imgs_dir ='./VOC2020/JPEGImages/'
    xmls_dir = './VOC2020/Annotations/'
    
    imgs_save_dir= './crop_512_imgs/'
    if not os.path.exists(imgs_save_dir):
        os.makedirs(imgs_save_dir)
    xmls_save_dir='./crop_512_xmls/'
    if not os.path.exists(xmls_save_dir):
        os.makedirs(xmls_save_dir)
    img_suffix=['.jpg','.png','.bmp']
    name_suffix='crop' #命名标识
    crop_type = 'TwentyFour_CROP'  
    crop_n = 24 #每张原图 crop 5张图
    dsize = (512,512) #指定裁剪尺度
    fw=1   
    fh=1  #指定裁剪尺度比例
    random_wh=False #是否随机尺度裁剪，若为True,则dsize指定的尺度失效
    iou_thr = 0.5 #裁剪后目标框大小与原框大小的iou值大于该阈值则保留
    crop_img_xml_from_dir(imgs_dir,xmls_dir,imgs_save_dir,xmls_save_dir,img_suffix,name_suffix,\
                                 crop_type,crop_n,dsize,fw,fh,random_wh,iou_thr)
#    crop_imgs_without_label(imgs_dir,imgs_save_dir,name_suffix,crop_type,\
#                            crop_n,dsize,fw,fh,random_wh)
    
if __name__=='__main__':
    main()        

   
    
    
    