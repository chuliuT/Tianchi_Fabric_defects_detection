# 广东工业智造大赛--赛场一 布匹瑕疵检测
##这是我第一次参加天池的大赛，半决赛的代码开源在了final_commit文件夹
##里面包含了填鸭的代码，第一版的填鸭，计算了patch块的相似度。第二版的我们对小目标（1-4）类的随机3-5倍的放大。
##半决赛的Rank：34/100
##线下的map：60%左右，线上的map：40%左右。
##总结一下：1.特征工程 2.选用模型 3.训练，调参 4.提交结果
##开始做的时候，我们是先出了一个baseline的结果，开始我是自己一个人玩，直接上faster-rcnn-r101 map：26% 有点沮丧
##毕竟这时候还在忙着写（水）论文，后来论坛里 开源了一个cascade-rcnn-r50的模型。初赛52%map。
##根据这个baseline，我换了backbone r101 居然：54%map，嘻嘻...这里直接就进了60多。
## 点1：anchor的设计非常重要，mmdetection的默认[0.5,1,2]一般来说很难符合数据的特性，所以这里是提分的点子
## 点2：fpn层 dcn （槽点，太吃显存，因为要用很多的offsets）
## 点3：OHEM 在线困难样本的发掘
## 点4：soft-nms 提分不多，大概一个点左右（大概率是0.几%哈~）

## 点5：TTA 老版本的mmdet没有TTA 多尺度测试，新版的有
## 点6：填鸭，对于正常样本的利用。这里其实跟我写论文里的东西有点像，来自小样本增强的那篇论文，但是有个问题就是
## 容易引入结构化的噪声，这里需要计算 patch的块的相似度---于是乎（度娘了一下），用了现成的
## 点7： GN+ws
## 点8： rpn 调参 


#一些试过但是没有用的
## 打算切图做的，cascade-r50对512*512的图像来测试，对于1-4好像有点效果，可能是可视化的错觉把

## 分类的模型，这时候就做的比较晚，用了 20层的 只有40多的acc

# 还没有尝试的
## GIOU loss
## GMloos GMHRLoss
## Focal loss
## mixup、smooth label
## 余弦退火学习率衰减


# 定制化的框架（可能是我没有仔细阅读过mmdet的源码）
## 其实需要魔改框架的，加入一些对小目标增强的结构

#  爆显存
# 参考上传的 transform.py 替换 mmdet的同名文件
# This my first time participate in TianChi challenge!
# first TestB Rank：85/2714

## Install
fellow the mmdetection install.md

## Data format
In my experiment COCO,VOC changed by yourself!

## COCO pretrained model transfer
the transfer code in check points   num_class should modified your class

notice!!! scale and ratios changed,I use a simple cat method for suitable model struct param avoid parameters initialize problem!

## Training
```python3 configs/cascade_rcnn_r101_fpn_1x_with_coco.py  --gpus 1 work_dir XXXXXX(your path to save model and train log)```

## Test
```python3 cascade_rcnn_r101_fpn_1x_test_coco.py```


