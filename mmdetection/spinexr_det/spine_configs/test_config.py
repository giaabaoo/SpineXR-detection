from mmcv import Config
import dataset

cfg = Config.fromfile('../configs/dcn/cascade_rcnn_r101_fpn_dconv_c3-c5_1x_coco.py')

from mmdet.apis import set_random_seed

# model settings

cfg.model.roi_head.bbox_head[0].num_classes = 7
cfg.model.roi_head.bbox_head[1].num_classes = 7
cfg.model.roi_head.bbox_head[2].num_classes = 7
# dataset settings
CLASSES = ['Osteophytes',
            'Other lesions',
            'Spondylolysthesis',
            'Disc space narrowing',
            'Vertebral collapse',
            'Foraminal stenosis',
            'Surgical implant']

# Modify dataset type and path
cfg.data_root = '/home/dhgbao/Machine_Learning/SpineXR/vindr-spinexr-dataset/'
cfg.dataset_type = "SpineXRDataset"

cfg.data.samples_per_gpu=8
cfg.data.workers_per_gpu=1

cfg.data.train.ann_file = cfg.data_root + 'annotations/COCO/train.json'
cfg.data.train.img_prefix = cfg.data_root + 'train_pngs/'
cfg.data.train.type = 'SpineXRDataset'

cfg.data.val.ann_file = cfg.data_root + 'annotations/COCO/test_all.json'
cfg.data.val.img_prefix = cfg.data_root + 'test_pngs/'
cfg.data.val.type = 'SpineXRDataset'


cfg.data.test.ann_file = cfg.data_root + 'annotations/COCO/test_all.json'
cfg.data.test.img_prefix = cfg.data_root + 'test_pngs/'
cfg.data.test.type = 'SpineXRDataset'

# cfg.load_from = '../checkpoints/vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco_20201027pth-6879c318.pth'
cfg.load_from = "spinexr_exps/lastest.pth"

cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)

# Set up working dir to save files and logs.
cfg.work_dir = './spinexr_exps'

# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 8 since we only use one GPU.
cfg.optimizer.lr = 0.02 / 8
cfg.lr_config.warmup = None
cfg.log_config.interval = 10

cfg.model.test_cfg.rcnn.score_thr = 0.5

# Change the evaluation metric since we use customized dataset.
# cfg.evaluation.metric = 'mAP'

# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)

init_kwargs={
        'project': 'spinexr',
        'entity': 'aibigdata',
        # 'tags': ['vfnet', 'resnet50_2x'] 
    }

cfg.log_config.hooks = [
    dict(type='TextLoggerHook'),
    dict(type='MMDetWandbHook',
        init_kwargs=init_kwargs,
        log_checkpoint=True,
        log_checkpoint_metadata=True,
        num_eval_images=100,
        bbox_score_thr=0.5)]


# We can initialize the logger for training and have a look
# at the final config used for training
# cfg.runner = dict(type='EpochBasedRunner', max_epochs=1)
cfg.evaluation = dict(classwise=True)
print(f'Config:\n{cfg.pretty_text}')

