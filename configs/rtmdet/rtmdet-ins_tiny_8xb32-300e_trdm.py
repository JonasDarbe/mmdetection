_base_ = './rtmdet-ins_s_8xb32-300e_coco.py'
checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth'  # noqa
model = dict(
    type='RTMDet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[127.5, 127.5, 127.5],
        std=[255, 255, 255],
        bgr_to_rgb=False,
        batch_augments=None),
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.167,
        widen_factor=0.375,
        out_indices=(1,2,3,4),
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True),
        init_cfg=dict(
            type='Pretrained', prefix='backbone.', checkpoint=checkpoint)),
    neck=dict(
        type='CSPNeXtPAFPN',
        in_channels=[48, 96, 192, 384],
        out_channels=96,
        num_csp_blocks=1,
        expand_ratio=0.5,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True)),    
    bbox_head=dict(
        type='RTMDetInsSepBNHead',
        in_channels=96,
        feat_channels=96,
        _delete_=True,
        num_classes=3,
        stacked_convs=2,
        share_conv=True,
        pred_kernel_size=1,
        mask_loss_stride=2,
        act_cfg=dict(type='SiLU', inplace=True),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        anchor_generator=dict(
            type='MlvlPointGenerator', 
            offset=0, 
            strides=[4, 8, 16, 32]
        ),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_mask=dict(
            type='DiceLoss', loss_weight=2.0, eps=5e-6, reduction='mean')),
    train_cfg=dict(
        assigner=dict(type='DynamicSoftLabelAssigner', topk=13),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100,
        mask_thr_binary=0.5),
)

input_image_size = 1280

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(type='Resize', scale=(input_image_size, input_image_size), keep_ratio=True),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1)),
    dict(type='PackDetInputs')
]

batch_size = 4
num_workers = 1
data_root = '/mnt/e/[TRM][026][001]Road-annotation_2024-01-24_07-39-44'

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    dataset=dict(
        type='TrdmDataset',
        data_root=data_root,
        ann_file='train.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=None)
)


test_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='Resize', scale=(input_image_size, input_image_size), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape','scale_factor'))
]

val_dataloader = dict(
    batch_size=batch_size, 
    num_workers=num_workers, 
    dataset=dict(
        type='TrdmDataset',
        data_root=data_root,
        ann_file='val.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=test_pipeline,
        backend_args=None)
)
test_dataloader = val_dataloader

train_cfg = dict(
    max_epochs=50,
    val_interval=50)

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root+'/val.json',
    metric=['bbox', 'segm'],
    backend_args=None)
test_evaluator = val_evaluator

# val_evaluator = dict(type='SemSegMetric', iou_metrics=['mIoU'])
# test_evaluator = val_evaluator

default_hooks = dict(
    checkpoint=dict(
        interval=5,
        max_keep_ckpts=3  # only keep latest 3 checkpoints
    ))