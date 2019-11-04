# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 08:15:38 2018
生成voc xml文档类
@author: zxl
"""

from xml.dom.minidom import Document
import xml.etree.ElementTree as ET
import os

def get_xml_tree(xmlfile):
    '''
    获取xml tree
    Args:
        xmlfile: xml文件路径
    return:
        tree:xml tree
    '''
    tree = ET.parse(xmlfile)
    return tree
    

class CreateXML():
    def __init__(self,img_name,img_w,img_h,img_channels):
        '''
        Args:
            img_name:图片名
            img_w,img_h,img_channels:图片宽、高、通道数
        '''
        self.img_name = img_name
        self.doc = Document()

        self.annotation = self.doc.createElement('annotation')
        self.doc.appendChild(self.annotation)
        

        folder = self.doc.createElement('folder')
        folder.appendChild(self.doc.createTextNode("AIA AUTO"))
        self.annotation.appendChild(folder)
        
        filename = self.doc.createElement('filename')
        filename.appendChild(self.doc.createTextNode(img_name))
        self.annotation.appendChild(filename)
        
        source = self.doc.createElement('source')
        database = self.doc.createElement('database')
        database.appendChild(self.doc.createTextNode('The AUTO Database'))
        anno = self.doc.createElement("annotation")
        anno.appendChild(self.doc.createTextNode("AUTO by zxl"))
        image = self.doc.createElement("image")
        image.appendChild(self.doc.createTextNode("flickr"))
        source.appendChild(database)
        source.appendChild(anno)
        source.appendChild(image)
        self.annotation.appendChild(source)
        
        sizeimage = self.doc.createElement('size')
        imagewidth = self.doc.createElement('width')
        imagewidth.appendChild(self.doc.createTextNode(str(img_w)))
        imageheight = self.doc.createElement('height')
        imageheight.appendChild(self.doc.createTextNode(str(img_h)))
        imagedepth = self.doc.createElement("depth")
        imagedepth.appendChild(self.doc.createTextNode(str(img_channels)))
        sizeimage.appendChild(imagewidth)
        sizeimage.appendChild(imageheight)
        sizeimage.appendChild(imagedepth)
        self.annotation.appendChild(sizeimage)
        
    def add_object_node(self,obj_name,xmin_v,ymin_v,xmax_v,ymax_v,truncated_v = 0,difficult_v=0):
        '''
        添加目标框节点
        obj_name:目标名
        xmin_v,ymin_v,xmax_v,ymax_v:目标框左上右上坐标
        truncated_v:截断程度
        difficult:困难程度
        '''
        obj= self.doc.createElement("object")   
        objname = self.doc.createElement("name")
        objname.appendChild(self.doc.createTextNode(obj_name))
        pose = self.doc.createElement("pose")
        pose.appendChild(self.doc.createTextNode("front"))
        truncated = self.doc.createElement("truncated")
        truncated.appendChild(self.doc.createTextNode(str(truncated_v)))
        difficult = self.doc.createElement('difficult')
        difficult.appendChild(self.doc.createTextNode(str(difficult_v)))
        obj.appendChild(objname)
        obj.appendChild(pose)
        obj.appendChild(truncated)
        obj.appendChild(difficult)
    
        bndbox = self.doc.createElement("bndbox")
        xmin = self.doc.createElement("xmin")
        ymin = self.doc.createElement("ymin")
        xmax = self.doc.createElement("xmax")
        ymax = self.doc.createElement("ymax")
        xmin.appendChild(self.doc.createTextNode(str(xmin_v)))
        ymin.appendChild(self.doc.createTextNode(str(ymin_v)))
        xmax.appendChild(self.doc.createTextNode(str(xmax_v)))
        ymax.appendChild(self.doc.createTextNode(str(ymax_v)))
        bndbox.appendChild(xmin)
        bndbox.appendChild(ymin)
        bndbox.appendChild(xmax)
        bndbox.appendChild(ymax)
        obj.appendChild(bndbox)
        self.annotation.appendChild(obj)
        
    def save_xml(self,save_path,xml_save_name):
        '''
        save_path:保存路径
        xml_save_name:xml文件保存名字       
        '''
        xml_file = open(os.path.join(save_path,xml_save_name),'w')
        xml_file.write(self.doc.toprettyxml(indent=' '*4))
        
    def get_doc(self):
        '''
        return:
            doc:xml文件的Document()
        '''
        return self.doc