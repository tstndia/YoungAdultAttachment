_base_ = [
    '../../_base_/models/swin/swin_small.py', '../../_base_/default_runtime.py'
]
model=dict(
    backbone=dict(
        patch_size=(2,4,4),
        drop_path_rate=0.1
    ),
    cls_head=dict(
        num_classes=8
    ),
    test_cfg=dict(
        max_testing_views=4
    )
)

# dataset settings
dataset_type = 'MultilabelVideoDataset'
data_root = 'data/response_video/'
data_root_val = 'data/response_video/'
ann_file_train = 'data/response_video/video_response_train.txt'
ann_file_val = 'data/response_video/video_response_val.txt'
ann_file_test = 'data/response_video/video_response_val.txt'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=64, frame_interval=16, num_clips=1),
    dict(type='DecordDecode'),
    #dict(type='Resize', scale=(-1, 64)),
    #dict(type='RandomResizedCrop'),
    #dict(type='Resize', scale=(64, 64), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=64,
        frame_interval=16,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    #dict(type='Resize', scale=(-1, 64)),
    #dict(type='CenterCrop', crop_size=64),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=64,
        frame_interval=16,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    #dict(type='Resize', scale=(-1, 64)),
    #dict(type='ThreeCrop', crop_size=64),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=2,
    workers_per_gpu=4,
    val_dataloader=dict(
        videos_per_gpu=1,
        workers_per_gpu=1
    ),
    test_dataloader=dict(
        videos_per_gpu=1,
        workers_per_gpu=1
    ),
    train=dict(
        type=dataset_type,
        multi_class=True,
        num_classes=8,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        multi_class=True,
        num_classes=8,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        multi_class=True,
        num_classes=8,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline))
evaluation = dict(
    interval=1, metrics=['confusion_matrix', 'mean_class_accuracy']
)

# optimizer
optimizer = dict(type='AdamW', lr=1e-3, betas=(0.9, 0.999), weight_decay=0.02,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'backbone': dict(lr_mult=0.1)}))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=2.5
)
total_epochs = 20

# runtime settings
checkpoint_config = dict(interval=1)
work_dir = './work_dirs/attachment_exposure_small_cl128_fi8_b8_ep20'
find_unused_parameters = False


# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)

#workflow = [('train', 1),  ('val', 1)]