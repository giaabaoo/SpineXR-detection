_base_ = ['./dynamic_rcnn_r50_fpn_1x_coco.py'] # kế thừa lại toàn bộ configs được viết sẵn

# model settings
model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=7)
    )
)

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

# train_pipeline = [
#     dict(type='TextLoggerHook'),
#     dict(type='MMDetWandbHook',
#         init_kwargs={'project': 'mmdetection'},
#         interval=10,
#         log_checkpoint=True,
#         log_checkpoint_metadata=True,
#         num_eval_images=100,
#         bbox_score_thr=0.3)
# ]
# test_pipeline = [
#     dict(type='TextLoggerHook'),
#     dict(type='MMDetWandbHook',
#         init_kwargs={'project': 'mmdetection'},
#         interval=10,
#         log_checkpoint=True,
#         log_checkpoint_metadata=True,
#         num_eval_images=100,
#         bbox_score_thr=0.3)
# ]

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
        
# cfg.log_config.hooks = [
#     dict(type='TextLoggerHook'),
#     dict(type='MMDetWandbHook',
#          init_kwargs={'project': 'mmdetection'},
#          interval=10,
#          log_checkpoint=True,
#          log_checkpoint_metadata=True,
#          num_eval_images=100,
#          bbox_score_thr=0.3)]
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size_divisor=32),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
# ]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(1333, 800),
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='Pad', size_divisor=32),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ])
# ]
# data = dict(
#     samples_per_gpu=2,
#     workers_per_gpu=2,
#     train=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/instances_train2017.json',
#         img_prefix=data_root + 'train2017/',
#         pipeline=train_pipeline),
#     val=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/instances_val2017.json',
#         img_prefix=data_root + 'val2017/',
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/instances_val2017.json',
#         img_prefix=data_root + 'val2017/',
#         pipeline=test_pipeline))
# evaluation = dict(interval=1, metric='bbox')