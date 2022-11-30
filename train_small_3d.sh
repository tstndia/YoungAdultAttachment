#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python tools/train.py $2 --cfg-options model.backbone.pretrained=pretrained/swin_small_patch244_window877_kinetics400_1k.pth model.backbone.pretrained2d=False