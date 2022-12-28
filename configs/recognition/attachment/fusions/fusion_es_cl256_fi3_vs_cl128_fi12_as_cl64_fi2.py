exposure = dict(
    config='configs/recognition/attachment/attachment_exposure_small_cl256_fi3.py',
    checkpoint='work_dirs/attachment_exposure_small_cl256_fi3_b8_ep30/latest.pth'
)

video_response = dict(
    config='configs/recognition/attachment/attachment_response_video_small_cl128_fi12.py',
    checkpoint='work_dirs/attachment_response_video_small_cl128_fi12_b8_ep30/latest.pth'
)

audio_response = dict(
    config='configs/recognition/attachment/attachment_response_audio_small_cl64_fi2.py',
    checkpoint='work_dirs/attachment_response_audio_small_cl64_fi2_ep100/latest.pth'
)