import mmcv
from mmcv.runner import load_checkpoint

from mmdet.apis import inference_detector, show_result_pyplot
from mmdet.models import build_detector
from inference_config import cfg
import numpy as np
import pdb

# Setup a checkpoint file to load
checkpoint = './spinexr_exps/latest.pth'

# Set the device to be used for evaluation
device='cuda:0'

# Initialize the detector
model = build_detector(cfg.model)

# Load checkpoint
checkpoint = load_checkpoint(model, checkpoint, map_location=device)

# Set the classes of models for inference
model.CLASSES = checkpoint['meta']['CLASSES']

# We need to set the model's cfg for inference
model.cfg = cfg

# Convert the model into evaluation mode
model.eval()

img = '/home/dhgbao/Machine_Learning/SpineXR/vindr-spinexr-dataset/test_pngs/3f503dc886d4105b86e199f6ef3bd722.png'
img = mmcv.imread(img)
result = inference_detector(model, img)

bbox_result = result 
num_samples_each_class = [bbox_result[i].shape[0] for i in range(len(bbox_result))]

# pdb.set_trace()
# bboxes = np.vstack(bbox_result)

abnormal_threshold = 0.51
abnormal_prob = 0.4
keep_threshold = 0.5

normal = abnormal_prob < abnormal_threshold

if normal:
    for i in range(7):
        # pdb.set_trace()
        for j, instance in enumerate(bbox_result[i]):
            if instance[-1] < keep_threshold: #score
                bbox_result[i][j] = np.array([0,0,0,0,0])


model.show_result(img, bbox_result,out_file="lesion.png")

# labels_impt = np.where(bboxes[:, -1] > 0.3)[0]

# left = bboxes[labels_impt][0][0]
# top = bboxes[labels_impt][0][1]
# right = bboxes[labels_impt][0][2]
# bottom = bboxes[labels_impt][0][3]
# 7 x 35 x 10
# labels = [
#     np.full(bbox.shape[0], i, dtype=np.int32)\
#     for i, bbox in enumerate(bbox_result)
# ]
# labels = np.concatenate(labels)

# classes = model.CLASSES
# labels_impt_list = [labels[i] for i in labels_impt]
# labels_class = [classes[i] for i in labels_impt_list]


