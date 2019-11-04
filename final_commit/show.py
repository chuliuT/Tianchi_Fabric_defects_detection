# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 14:50:25 2018
显示
@author: zxl
"""

import cv2
import os 
import utils
import math

import xml.etree.ElementTree as ET

def get_color_channel(c,offset,maxclass):
    '''获取每个通道的颜色值
    Args:
        c:颜色通道
        offset:类别偏置
        maxclass:最大类别数
    return:
        r:该通道颜色
    '''
    colors = [[1,0,1],[0,0,1],[0,1,1],[0,1,0],[1,1,0],[1,0,0]]
    ratio = (offset/maxclass)*5
    i = math.floor(ratio)
    j = math.ceil(ratio)
    ratio -= i
    r = (1-ratio)*colors[i][c]+ratio*colors[j][c]
    return r

def get_color(cls,maxcls=20):
    '''为一个类别生成一种特定显示颜色
    Args:
        cls:类别id (from 0)
        maxcls:最大类别数
    return:
        color:(B,G,R) 颜色
    '''
    if cls > maxcls:
        maxcls = maxcls*(int(cls/maxcls) + 1)
    offset = cls*123457%maxcls
    b = get_color_channel(0,offset,maxcls)*255
    g = get_color_channel(1,offset,maxcls)
    r = get_color_channel(2,offset,maxcls)
    return (int(b*255),int(g*255),int(r*255))
    

def show_data(img_file,xml_file,windowname='ORG',class_color={},showname=True,maxcls=20,wait_sec=0):
    '''显示一张图片
    Args:
        img_file:图片文件
        xml_file:xml标注文件
        windowname:显示窗口名
        class_color:已有类别目标框显示颜色
        showname:是否显示类别名
        maxcls:最大类别
        wait_sec:opencv响应等待时间
    return:
        key:opencv响应键值
    '''
    tree = ET.parse(xml_file)
    xml_root = tree.getroot()
    
    cv2.namedWindow(windowname,cv2.WINDOW_AUTOSIZE)
    
    img = cv2.imread(img_file)
    rows,cols,_ = img.shape
    
    for obj in xml_root.iter('object'):
        cls_name = obj.find('name').text
        if cls_name in class_color:
            color = class_color[cls_name]
        else:
            cls_id = len(class_color)
            color = get_color(cls_id,maxcls)
            class_color[cls_name] = color
                           
        xmlbox = obj.find('bndbox')
        box = list(map(int,[float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text), \
                                    float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text)]))   
        cv2.rectangle( img,(box[0],box[1]),(box[2],box[3]),color,max([int(min([rows,cols])*0.003),1]))
        if showname:
            retval,baseline = cv2.getTextSize(cls_name,cv2.FONT_HERSHEY_PLAIN,\
                                                      0.1*rows/90,max([int(min([rows,cols])*0.001),1]))
            cv2.rectangle(img,(box[0],box[1]-retval[1]),(box[0]+retval[0],box[1]),color,-1,8,0)
            cv2.putText(img,cls_name,(box[0],box[1]),cv2.FONT_HERSHEY_PLAIN,0.1*rows/90,\
                                (0,0,0),max([int(min([rows,cols])*0.001),1]))    
    cv2.imshow(windowname,img)
    cv2.imwrite('./xml_image.jpg',img)
    key = cv2.waitKeyEx(wait_sec) #waitKey对上下左右方向键的返回值均为0，waitKeyEx有不同的值
    return key
    


def show_data_in_dir(imgs_dir,xmls_dir,windowname='ORG',class_color={},showname=True,maxcls=20,delete=False):
    '''显示图片和标注框
    Args:
        imgs_dir:图片目录
        xmls_dir:标注文件xml目录，voc格式
        windowname：显示窗口名
        class_color：类别显示颜色的BGR值
        showname:是否显示类别名
        maxcls:最大类别
        delete:是否删除没有图片的xml文件
    '''
    
    xml_count,img_count = utils.fileCountIn(xmls_dir),utils.fileCountIn(imgs_dir)
    print('------show object boxes based on xml files (xml:%d,image:%d)------'%(xml_count,img_count))
    count = 0
    
    cv2.namedWindow(windowname,cv2.WINDOW_NORMAL)
    wait_sec = 0
    
    for root,dirs,files in os.walk(xmls_dir):
        idx = 0
        while idx < len(files):
            file = files[idx]
            count += 1
            if count%100 == 0:
                print('[%d | %d]%d%%'%(xml_count,count,count*100/xml_count))
            
            xml_file = os.path.join(xmls_dir,file)
            tree = ET.parse(xml_file)
            xml_root = tree.getroot()
            img_name = xml_root.find('filename').text
            
            img_file = os.path.join(imgs_dir,img_name)
            if not os.path.exists(img_file):
                print('%s not exist!'%img_file)
                if delete:
                    os.remove(xml_file)
                    print(xml_file,'has been removed!')
                    idx += 1
                continue
            print(img_name)
            key = show_data(img_file,xml_file,windowname,class_color,showname,maxcls,wait_sec)
            
            if(32==key):
                wait_sec = 1-wait_sec
            elif(key==ord('q') or key==ord('Q') ):
                return 0
            elif(key==2424832 or key==2490368 or key == ord('p')):
                #左、上方向键或p查看上一张图片
                idx -= 1
            else:
                idx += 1

    cv2.destroyAllWindows()
    return 0

def show_data_in_pathfile(pathfile,windowname='ORG',class_color={},showname=True,maxcls=20):
    '''根据pathfile文件中的图片路径显示图片和标注框，要求以voc标准格式存放
    Args:
        pathfile：图片路径文件
        windowname：显示窗口名
        class_color：类别颜色的RGB值
        showname:是否显示类别名
        maxcls:最大类别
    '''    
    imgpathfiles = open(pathfile)
    imgfilelines = imgpathfiles.readlines()
    fileCount = len(imgfilelines)
    print("----------- %d images------------"%fileCount)

    count = 0    
    cv2.namedWindow(windowname,cv2.WINDOW_AUTOSIZE)
    wait_sec = 0
    
    idx = 0
    while idx < fileCount:
        imgfile = imgfilelines[idx].strip()
        dirname = os.path.dirname(imgfile).replace('JPEGImages','Annotations')
        xmlfile = os.path.join(dirname,os.path.basename(imgfile).split('.')[0]+'.xml')
        
        count += 1
        if count%100 == 0:
            print('[%d | %d]%d%%'%(fileCount,count,count*100/fileCount))
            
        
        if not os.path.exists(xmlfile):
            print(xmlfile,' not exist!')
            idx += 1
            continue
        if not os.path.exists(imgfile):
            print(imgfile,' not exist')
            idx += 1
            continue
        
        print(os.path.basename(imgfile))
        key = show_data(imgfile,xmlfile,windowname,class_color,showname,maxcls,wait_sec)
        
        if(32==key):
            wait_sec = 1-wait_sec
        elif(key==ord('q') or key==ord('Q') ):
            return 0
        elif(2424832==key or 2490368==key or key==ord('p')):
            #左、上方向键或p查看上一张图片
            idx -= 1
        else:
            idx += 1
    cv2.destroyAllWindows()        
    

           
def main():
    # img_file = './paste_test/JPEGImages/60639D0F.jpg'
    # xml_file = './paste_test/Annotations/60639D0F.xml'
#    imgs_dir = 'C:/Users/zxl/Desktop/test/JPEGImages/'
#    xmls_dir = 'C:/Users/zxl/Desktop/test/Annotations/'
    imgs_dir = './crop_512_imgs/'
    xmls_dir = './crop_512_xmls/'
    
    # imgpathsfile = 'E:/myjob/DataSet/DETRAC_VOC_v2/detrac_train_v2.txt'
    
    #show_data(img_file,xml_file) #显示单张图片标注框
    
    #显示文件夹中的图片和标注文件
    #空格键连续显示，左、上键显示上一张，右、下键显示下一张，q键退出
    show_data_in_dir(imgs_dir,xmls_dir,showname=True,maxcls=20)
    
    #显示路径文件中的图片和标注文件（voc标准格式）
    #空格键连续显示，左、上键和p 显示上一张，右、下键显示下一张，q键退出
    #show_data_in_pathfile('./paste_test/')
    
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()    
             