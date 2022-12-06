_base_ = [
    '../../_base_/models/tsn_r50_audio.py', '../../_base_/default_runtime.py'
]
model=dict(
    cls_head=dict(
        num_classes=8,
        loss_cls=dict(type='BCELossWithLogits', loss_weight=160.0),
        multi_class=False
    )
)

clip_len = 128
frame_interval = 3

# dataset settings
dataset_type = 'AudioDataset'
data_root = 'data/response_audio/'
data_root_val = 'data/response_audio/'
ann_file_train = 'data/response_audio/audio_response_train.txt'
ann_file_val = 'data/response_audio/audio_response_val.txt'
ann_file_test = 'data/response_audio/audio_response_val.txt'

train_pipeline = [
    dict(type='AudioDecodeInit'),
    dict(type='SampleFrames', clip_len=clip_len, frame_interval=frame_interval, num_clips=1),
    dict(type='AudioDecode'),
    dict(type='AudioAmplify', ratio=1.5),
    dict(type='MelSpectrogram'),
    dict(type='FormatAudioShape', input_format='NCTF'),
    dict(type='Collect', keys=['audios', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['audios'])
]
val_pipeline = [
    dict(type='AudioDecodeInit'),
    dict(
        type='SampleFrames', clip_len=clip_len, frame_interval=frame_interval, num_clips=1, 
        test_mode=True),
    dict(type='AudioDecode'),
    dict(type='AudioAmplify', ratio=1.5),
    dict(type='MelSpectrogram'),
    dict(type='FormatAudioShape', input_format='NCTF'),
    dict(type='Collect', keys=['audios', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['audios'])
]
test_pipeline = [
    dict(type='AudioDecodeInit'),
    dict(
        type='SampleFrames', clip_len=clip_len, frame_interval=frame_interval, num_clips=1,
        test_mode=True),
    dict(type='AudioDecodeInit'),
    dict(type='AudioAmplify', ratio=1.5),
    dict(type='MelSpectrogram'),
    dict(type='FormatAudioShape', input_format='NCTF'),
    dict(type='Collect', keys=['audios', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['audios'])
]
data = dict(
    videos_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        multi_class=True,
        num_classes=8,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        multi_class=True,
        num_classes=8,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        multi_class=True,
        num_classes=8,
        data_prefix=data_root_val,
        pipeline=test_pipeline))
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# optimizer
optimizer = dict(
    type='SGD', lr=0.1, momentum=0.9,
    weight_decay=0.0001)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0)
total_epochs = 30

# runtime settings
checkpoint_config = dict(interval=5)
work_dir = './work_dirs/attachment_response_audio_cl64_fi16/'