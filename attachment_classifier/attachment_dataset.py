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
        # print(exposure.shape)
        video = data[8*14:2*8*14]
        # print(video.shape)
        audio = data[2*8*14:3*8*14]
        # print(audio.shape)
        quiz = data[3*8*14:]
        # print(quiz.shape)

        data2 = torch.zeros(3 * 8 * 14 + 36, dtype=torch.float)

        if "e" in self.modality:
            data2[0:8*14] = exposure
        if "v" in self.modality:
            data2[8*14:2*8*14] = video
        if "a" in self.modality:
            data2[2*8*14:3*8*14] = audio
        if "q" in self.modality:
            data2[3*8*14:] = quiz

        # if self.modality == 'exposure':
        #     data = exposure
        # elif self.modality == 'video_response':
        #     data = video
        # elif self.modality == 'audio_response':
        #     data = audio
        # else:
        #     data = torch.stack([exposure, video, audio], dim=0).sum(dim=0)
        #     print(data.shape)
        #
        # if self.modality is None:
        #     data = torch.cat([data, quiz], dim=0)
        # data = normalize(data)

        # one hot
        tlabel = torch.zeros(self.num_classes, dtype=torch.float)
        tlabel[label] = 1.

        return data2, tlabel
            
    def __len__(self):
        return len(self.df)