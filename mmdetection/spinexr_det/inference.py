import mmcv
from mmcv.runner import load_checkpoint

from mmdet.apis import inference_detector, show_result_pyplot
from mmdet.models import build_detector
from spine_configs.inference_config import cfg

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

model.show_result(img, result,score_thr=0.5 ,out_file="lesion.png")

