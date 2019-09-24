# This my first time participate in TianChi chanllege!
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


