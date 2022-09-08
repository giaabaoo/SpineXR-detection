_base_ = ['./vfnet_r50_fpn_1x_coco.py'] 

# model settings
model = dict(
    type='VFNet',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='VFNetHead',
        num_classes=7,
        in_channels=256,
        stacked_convs=3,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        center_sampling=False,
        dcn_on_last_conv=False,
        use_atss=True,
        use_vfl=True,
        loss_cls=dict(
            type='VarifocalLoss',
            use_sigmoid=True,
            alpha=0.75,
            gamma=2.0,
            iou_weighted=True,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.5),
        loss_bbox_refine=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

# dataset settings
CLASSES = ['Osteophytes',
            'Other lesions',
            'Spondylolysthesis',
            'Disc space narrowing',
            'Vertebral collapse',
            'Foraminal stenosis',
            'Surgical implant']

dataset_type = 'CocoDataset'
data_root = '/home/dhgbao/Machine_Learning/SpineXR/vindr-spinexr-dataset/'

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/COCO/train.json',
        img_prefix=data_root + 'train_pngs/',
        classes = CLASSES),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/COCO/test.json',
        img_prefix=data_root + 'test_pngs/',
        classes = CLASSES),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/COCO/test.json',
        img_prefix=data_root + 'test_pngs/',
        classes = CLASSES))

runner = dict(type='EpochBasedRunner', max_epochs=20)
