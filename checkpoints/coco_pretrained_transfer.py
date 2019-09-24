# def main():
#     #gen coco pretrained weight
#     import torch
#     num_classes = 21
#     model_coco = torch.load("cascade_rcnn_r101_fpn_1x_20181129-d64ebac7.pth")

#     # weight
#     model_coco["state_dict"]["bbox_head.0.fc_cls.weight"] = model_coco["state_dict"]["bbox_head.0.fc_cls.weight"][
#                                                             :num_classes, :]
#     model_coco["state_dict"]["bbox_head.1.fc_cls.weight"] = model_coco["state_dict"]["bbox_head.1.fc_cls.weight"][
#                                                             :num_classes, :]
#     model_coco["state_dict"]["bbox_head.2.fc_cls.weight"] = model_coco["state_dict"]["bbox_head.2.fc_cls.weight"][
#                                                             :num_classes, :]
#     # bias
#     model_coco["state_dict"]["bbox_head.0.fc_cls.bias"] = model_coco["state_dict"]["bbox_head.0.fc_cls.bias"][
#                                                           :num_classes]
#     model_coco["state_dict"]["bbox_head.1.fc_cls.bias"] = model_coco["state_dict"]["bbox_head.1.fc_cls.bias"][
#                                                           :num_classes]
#     model_coco["state_dict"]["bbox_head.2.fc_cls.bias"] = model_coco["state_dict"]["bbox_head.2.fc_cls.bias"][
#                                                           :num_classes]
#     # model_coco["state_dict"]["rpn_head.rpn_cls.weight"] =model_coco["state_dict"]["rpn_head.rpn_cls.weight"]
#     # save new model
#     #torch.save(model_coco, "cascade_rcnn_r101_coco_pretrained_weights_classes_%d.pth" % num_classes)

# if __name__ == "__main__":
#     main()


# for cascade rcnn
import torch
num_classes = 21
model_coco = torch.load("cascade_rcnn_r101_fpn_1x_20181129-d64ebac7.pth")
# weight
model_coco["state_dict"]["bbox_head.0.fc_cls.weight"].resize_(num_classes,1024)
model_coco["state_dict"]["bbox_head.1.fc_cls.weight"].resize_(num_classes,1024)
model_coco["state_dict"]["bbox_head.2.fc_cls.weight"].resize_(num_classes,1024)
# bias
model_coco["state_dict"]["bbox_head.0.fc_cls.bias"].resize_(num_classes)
model_coco["state_dict"]["bbox_head.1.fc_cls.bias"].resize_(num_classes)
model_coco["state_dict"]["bbox_head.2.fc_cls.bias"].resize_(num_classes)
model_coco["state_dict"]["rpn_head.rpn_cls.weight"]=torch.cat([model_coco["state_dict"]["rpn_head.rpn_cls.weight"],model_coco["state_dict"]["rpn_head.rpn_cls.weight"],model_coco["state_dict"]["rpn_head.rpn_cls.weight"]],dim=0)
model_coco["state_dict"]["rpn_head.rpn_cls.bias"]=torch.cat([model_coco["state_dict"]["rpn_head.rpn_cls.bias"],model_coco["state_dict"]["rpn_head.rpn_cls.bias"],model_coco["state_dict"]["rpn_head.rpn_cls.bias"]],dim=0)
model_coco["state_dict"]["rpn_head.rpn_reg.weight"]=torch.cat([model_coco["state_dict"]["rpn_head.rpn_reg.weight"],model_coco["state_dict"]["rpn_head.rpn_reg.weight"],model_coco["state_dict"]["rpn_head.rpn_reg.weight"]],dim=0)
model_coco["state_dict"]["rpn_head.rpn_reg.bias"]=torch.cat([model_coco["state_dict"]["rpn_head.rpn_reg.bias"],model_coco["state_dict"]["rpn_head.rpn_reg.bias"],model_coco["state_dict"]["rpn_head.rpn_reg.bias"]],dim=0)

#save new model
torch.save(model_coco,"coco_pretrained_weights_classes_%d.pth"%num_classes)
