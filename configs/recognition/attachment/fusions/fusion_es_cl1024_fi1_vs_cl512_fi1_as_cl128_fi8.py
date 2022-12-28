exposure = dict(
    config='configs/recognition/attachment/attachment_exposure_small_cl1024_fi1.py',
    checkpoint='work_dirs/attachment_exposure_small_cl1024_fi1_b8_ep30/latest.pth'
)

video_response = dict(
    config='configs/recognition/attachment/attachment_response_video_small_cl512_fi1.py',
    checkpoint='work_dirs/attachment_response_video_small_cl512_fi1_b8_ep30/latest.pth'
)

audio_response = dict(
    config='configs/recognition/attachment/attachment_response_audio_small_cl128_fi8.py',
    checkpoint='work_dirs/attachment_response_audio_small_cl128_fi8_ep100/latest.pth'
)