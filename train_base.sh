#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python tools/train.py $2 --cfg-options model.backbone.pretrained=pretrained/swin_base_patch4_window7_224.pth