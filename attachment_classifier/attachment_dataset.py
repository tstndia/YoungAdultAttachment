import os
import pathlib
import collections
import numpy as np
import torch
import torch.utils.data
import cv2  # pytype: disable=attribute-error
import random
import pandas as pd

from torch.nn.functional import one_hot, normalize, pad

class AttachmentDataset(torch.utils.data.Dataset):
    def __init__(self, root, csv_file, num_classes, modality):
        self.root = root
        self.num_classes = num_classes
        self.modality = modality

        if not os.path.exists(root):
            raise ValueError("Path does not exist: " + root)

        self.df = pd.read_csv(csv_file)
            
    def __getitem__(self, index):
        item = self.df.iloc[index].to_dict()
        filename, label = item['filename'], item['label']
        
        #data = np.zeros(336)
        #attachment = np.load(os.path.join(self.root, filename))
        #data[0:len(attachment)] = attachment
        data = np.load(os.path.join(self.root, filename))

        data = torch.from_numpy(data).type(torch.float)
        exposure = data[0:8*14]
        video = data[8*14:2*8*14]
        audio = data[2*8*14:3*8*14]
        quiz = data[3*8*14:]
        
        if self.modality == 'exposure':
            data = exposure
        elif self.modality == 'video_response':
            data = video
        elif self.modality == 'audio_response':
            data = audio
        else:
            data = torch.stack([exposure,video,audio],dim=0).sum(dim=0)
        # elif self.modality == 'exp-respv-sra':
        #     data = torch.stack([exposure, video, audio], dim=0).sum(dim=0)
        # elif self.modality == 'exp-respv-quest':
        #     data = torch.stack([exposure,video],dim=0).sum(dim=0)
        #     data = torch.cat([data,quiz],dim=0)
        # elif self.modality == 'exp-sra-quest':
        #     data = torch.stack([exposure, audio], dim=0).sum(dim=0)
        #     data = torch.cat([data, quiz], dim=0)
        # elif self.modality == 'respv-sra-quest':
        #     data = torch.stack([video, audio], dim=0).sum(dim=0)
        #     data = torch.cat([data, quiz], dim=0)
        # elif self.modality == 'exp-quest':
        #     data = torch.cat([exposure,quiz],dim=0)
        # elif self.modality == 'respv-quest':
        #     data = torch.cat([video,quiz],dim=0)
        # elif self.modality == 'sra-quest':
        #     data = torch.cat([audio,quiz],dim=0)
        # elif self.modality == 'exp-respv':
        #     data = torch.stack([exposure,video],dim=0).sum(dim=0)
        # elif self.modality == 'exp-sra':
        #     data = torch.stack([exposure,audio],dim=0).sum(dim=0)
        # elif self.modality == 'respv-sra':
        #     data = torch.stack([video,audio],dim=0).sum(dim=0)
        # else:
        #     data = quiz
        
        if self.modality is None:
            data = torch.cat([data, quiz], dim=0)
        # data = normalize(data)

        # one hot
        tlabel = torch.zeros(self.num_classes, dtype=torch.float)
        tlabel[label] = 1.

        return data, tlabel
            
    def __len__(self):
        return len(self.df)