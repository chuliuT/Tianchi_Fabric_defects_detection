import time, os
import json
import mmcv 
from mmdet.apis import init_detector, inference_detector,show_result

def main():

    config_file = 'configs/cascade_rcnn_r101_fpn_1x.py'  # 修改成自己的配置文件
    checkpoint_file = '/home/titan/mmdetection/tianchi_round1_cascade_r101_with_coco/latest.pth' # 修改成自己的训练权重

    test_path = 'data/TianChi_round1/guangdong1_round1_testA_20190818'  # 官方测试集图片路径

    json_name = "result_"+""+time.strftime("%Y%m%d%H%M%S", time.localtime())+".json"
    
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    img_list = []
    for img_name in os.listdir(test_path):
        if img_name.endswith('.jpg'):
            img_list.append(img_name)

    result = []
    for i, img_name in enumerate(img_list, 1):
        full_img = os.path.join(test_path, img_name)
        predict = inference_detector(model, full_img)
        show_result(full_img, predict, model.CLASSES,out_file='./coco_soft_nms_result/result_{}'.format(full_img))
        for i, bboxes in enumerate(predict, 1):
            if len(bboxes)>0:
                defect_label = i
                print(i)
                image_name = img_name
                for bbox in bboxes:
                    x1, y1, x2, y2, score = bbox.tolist()
                    x1, y1, x2, y2 = round(x1,2), round(y1,2), round(x2,2), round(y2,2)  #save 0.00
                    result.append({'name': image_name, 'category': defect_label, 'bbox': [x1,y1,x2,y2], 'score': score})

    with open(json_name,'w') as fp:
        json.dump(result, fp, indent = 4, separators=(',', ': '))
        
if __name__ == "__main__":
    main()