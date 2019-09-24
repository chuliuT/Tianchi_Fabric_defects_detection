
import os
import cv2
import re
#原文：https://blog.csdn.net/angelbeats11/article/details/88427314 
pattens = ['name','xmin','ymin','xmax','ymax']
import matplotlib.pyplot as plt
import numpy as np
count=0
def get_annotations(xml_path):
    bbox = []
    with open(xml_path,'r') as f:
        text = f.read().replace('\n','return')
        p1 = re.compile(r'(?<=<object>)(.*?)(?=</object>)')
        result = p1.findall(text)
        for obj in result:
            tmp = []
            for patten in pattens:
                p = re.compile(r'(?<=<{}>)(.*?)(?=</{}>)'.format(patten,patten))
                if patten == 'name':
                    tmp.append(p.findall(obj)[0])
                else:
                    tmp.append(int(float(p.findall(obj)[0])))
            bbox.append(tmp)
    return bbox
 
def save_viz_image(image_path,xml_path,save_path):
    global count
    bbox = get_annotations(xml_path)
    image = cv2.imread(image_path)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for info in bbox:
        # print(info[0])

        count+=1
        # print(os.path.join(save_path,str(count),image_path.split('/')[-1]))
        # print(image[info[2]:info[4],info[1]:info[3]].shape)
        # plt.imshow(image[info[2]:info[4],info[1]:info[3]])
        # plt.show()

        # print(info[1],info[2],info[3],info[4])
        # cv2.rectangle(image,(info[1],info[2]),(info[3],info[4]),(0,255,0),thickness=2)
        # cv2.putText(image,info[0],(info[1],info[2]),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
        # gray=cv2.cvtColor(image[info[2]:info[4],info[1]:info[3]],cv2.COLOR_BGR2GRAY)
        # # ret2, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # im = np.float32(gray) / 255.0

        # # Calculate gradient
        # gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=1)
        # gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=1)
        # mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        #
        # plt.imshow(mag)
        # plt.show()
        # print(save_path+str(count)+'.jpg')
        cv2.imwrite(save_path+"/"+str(count)+'_'+str(info[0])+'.jpg',image[info[2]:info[4],info[1]:info[3]])
        #cv2.imwrite(save_path + str(count) + '.jpg', mag)
 
if __name__ == '__main__':
    image_dir = './VOC2020/JPEGImages'
    xml_dir = './VOC2020/Annotations'
    save_dir = './detect_images'
    image_list = os.listdir(image_dir)
    for i in  image_list:
        image_path = os.path.join(image_dir,i)
        xml_path = os.path.join(xml_dir,i.replace('.jpg','.xml'))
        save_viz_image(image_path,xml_path,save_dir)


