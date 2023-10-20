exposure = dict(
    config='configs/recognition/attachment/exposure_base_cl512_fi2.py',
    checkpoint='work_dirs/attachment_exposure_base_cl512_fi2_b8_ep30/latest.pth'
)

video_response = dict(
    config='configs/recognition/attachment/attachment_response_video_small_cl512_fi2.py',
    checkpoint='work_dirs/attachment_response_video_small_cl512_fi2_b8_ep30/latest.pth'
)

audio_response = dict(
    config='configs/recognition/attachment/attachment_response_audio_small_cl256_fi4.py',
    checkpoint='work_dirs/attachment_response_audio_small_cl256_fi4_ep100/latest.pth'
)