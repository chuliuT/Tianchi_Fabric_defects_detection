<<<<<<< HEAD
ARG PYTORCH="1.1.0"
ARG CUDA="10.0"
ARG CUDNN="7.5"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN conda install cython -y && conda clean --all

ADD ./mmdetection /competition/mmdetection
ADD ./mmdetection/configs /competition/configs
ADD ./mmdetection/round2_cascade_r50 /competition/round2_cascade_r50
ADD ./mmdetection/cascade_rcnn_r50_fpn_1x.py /competition/cascade_rcnn_r50_fpn_1x.py
ADD ./mmdetection/cascade_rcnn_r50_round2_test.py /competition/cascade_rcnn_r50_round2_test.py
ADD ./Dockerfile /competition/Dockerfile
ADD ./mmdetection/run.sh /competition/run.sh


WORKDIR /competition
RUN pip --no-cache-dir install -r ./mmdetection/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple  
RUN pip install --no-cache-dir -e ./mmdetection/.
CMD ["sh", "run.sh"]
=======
# Base Images
FROM registry.cn-shenzhen.aliyuncs.com/lk0703/mmdet_cloud:1.1.1-cuda10.0-py3
ADD . /
WORKDIR /
CMD ["sh", "run.sh"]
>>>>>>> ac37e5054aaddab019740397aa04eb1f0f0f6523
