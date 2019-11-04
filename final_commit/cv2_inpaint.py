import cv2
import numpy as np

bear_image=cv2.imread('./timg.jpg')
rescale_bear=cv2.resize(bear_image,(64,64))
# cv2.imshow('rescale_bear',rescale_bear)
# cv2.waitKey(0)
# print(rescale_bear.shape)

left_top_point=[100,200]
h,w,c=rescale_bear.shape


girl_image=cv2.imread('./girl.jpg')
# h,w,c=girl_image.shape
ring_width=5
# print(h,w,c)
mask=np.zeros_like(girl_image)
mask[left_top_point[0]-ring_width:left_top_point[0]+h+ring_width,left_top_point[1]-ring_width:left_top_point[1]+w+ring_width]=255
mask[left_top_point[0]:left_top_point[0]+h,left_top_point[1]:left_top_point[1]+w]=0
girl_image[left_top_point[0]:left_top_point[0]+h,left_top_point[1]:left_top_point[1]+w]=rescale_bear
cv2.imshow('mask',mask)
cv2.waitKey(0)
print(mask[:,:,0])
dst_TELEA = cv2.inpaint(girl_image,mask[:,:,0],3,cv2.INPAINT_TELEA)
cv2.imshow('dst_TELEA',dst_TELEA)
cv2.waitKey(0)