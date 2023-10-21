exposure = dict(
    config='configs/recognition/attachment/exposure_base_cl128_fi4.py',
    checkpoint='work_dirs/attachment_exposure_base_cl128_fi4_b8_ep30/latest.pth'
)

video_response = dict(
    config='configs/recognition/attachment/response_video_base_cl256_fi3.py',
    checkpoint='work_dirs/response128_video_base_cl256_fi3_b8_ep30/latest.pth'
)

audio_response = dict(
    config='configs/recognition/attachment/attachment_response_audio_small_cl256_fi2.py',
    checkpoint='work_dirs/attachment_response_audio_small_cl256_fi2_ep100/latest.pth'
)