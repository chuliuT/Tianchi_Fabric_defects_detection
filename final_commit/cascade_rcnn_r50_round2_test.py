import time, os
import json
import mmcv
from mmdet.apis import init_detector, inference_detector,show_result,show_result_pyplot
import numpy as np
# import pandas as pd
import cv2

left_top_point=[[0,0],[0,512],[0,1024],[512,0],[512,512],[512,1024],[1024,0],[1024,512],[1024,1024],[1536,0],[1536,512],[1536,1024],
[2560,0],[2560,512],[2560,1024],[3072,0],[3072,512],[3072,1024],[3584,0],[3584,512],[3584,1024]]

crop_w,crop_h=512,512

def crop_img(src,top_left_x,top_left_y,crop_w,crop_h):
    rows,cols,n_channel = src.shape
    row_min,col_min = int(top_left_y), int(top_left_x)
    row_max,col_max = int(row_min + crop_h), int(col_min + crop_w)
    crop_img = src[row_min:row_max,col_min:col_max]
    return crop_img


def main():

    config_file =  './cascade_rcnn_r50_fpn_1x.py'   # 修改成自己的配置文件
    checkpoint_file = './round2_cascade_r50/latest.pth'  # 修改成自己的训练权重

    test_path = '/tcdata/guangdong1_round2_testA_20190924'  # 官方测试集图片路径
    
    # json_name = "result_"+""+time.strftime("%Y%m%d%H%M%S", time.localtime())+".json"
    json_name = "result"+".json"
    
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    result = []
    for root,image_paths,image_files in os.walk(test_path):
        for path in image_paths:
            image_path=os.path.join(root,path,path+'.jpg')
            json_image_name=path+'.jpg'
            img=cv2.imread(image_path)
            for idx, pts in enumerate(left_top_point):
                [top_left_x,top_left_y]=pts
                crop_img2=crop_img(img,top_left_x,top_left_y,crop_w,crop_h)
                predict = inference_detector(model, crop_img2)
                for i, bboxes in enumerate(predict, 1):
                    if len(bboxes)>0:
                        defect_label = i
                        for bbox in bboxes:
                            x1, y1, x2, y2, score = bbox.tolist()
                            x1, y1, x2, y2 = round(x1+top_left_x,2), round(y1+top_left_y,2), round(x2+top_left_x,2), round(y2+top_left_y,2)  #save 0.00
                            result.append({'name': json_image_name, 'category': defect_label, 'bbox': [x1,y1,x2,y2], 'score': score})
    with open(json_name,'w') as fp:
        json.dump(result, fp, indent = 4, separators=(',', ': '))

        
if __name__ == "__main__":
    main()
