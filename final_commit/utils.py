# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 10:30:26 2018
实用工具
@author: zxl
"""
import os
import random

def confine(value,v_min,v_max):
    '''
    值的边界限制
    Args:
        value:输入值
        v_min,v_max:最大最小边界
    return:
        value:限制值
    '''
    value = v_min if value < v_min else value
    value = v_max if value > v_max else value
    return value


def fileCountIn(dir):
    '''
    计算文件夹下文件个数
    Args:
        dir:文件目录
    return:
        文件个数
    '''
    return sum([len(files) for root,dirs,files in os.walk(dir)])

def randomChoiceIn(dir):
    '''
    目录下随机选择一个文件
    Args:
        dir:目录
    return:
        filename:随机选择的文件名
    '''
    for root,dirs,files in os.walk(dir):
        index=random.randint(0,len(files)-1)
        filename = files[index]
    return filename


def calc_rect_area( rect ):
    '''计算矩形框面积
    Args:
        rect:矩形框 [xmin,ymin,xmax,ymax]
    return:
        dst:矩形框面积
    '''
    return (rect[2]-rect[0]+0.001)*(rect[3]-rect[1]+0.001)

def calc_iou(rect1,rect2):
    '''计算两个矩形框的交并比
    Args:
        rect1,rect2:两个矩形框
    return:
        iou:交并比
    '''
    bd_i= ( max(rect1[0],rect2[0]),max(rect1[1],rect2[1]),\
           min(rect1[2],rect2[2]),min(rect1[3],rect2[3]))
    iw = bd_i[2]-bd_i[0]+0.001
    ih = bd_i[3]-bd_i[1]+0.001
    iou=0
    if( iw>0 and ih>0 ):
        ua = calc_rect_area( rect1 ) + calc_rect_area( rect2 ) - iw*ih
        iou = iw*ih/ua		
    return iou


