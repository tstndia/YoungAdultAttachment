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
    def __init__(self, root, csv_file, num_classes):
        self.root = root
        self.num_classes = num_classes

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
        
        data = torch.stack([exposure, video, audio], dim=0).sum(dim=0)
        data = torch.cat([data, quiz], dim=0)
        #data = normalize(data)

        # one hot
        tlabel = torch.zeros(self.num_classes, dtype=torch.float)
        tlabel[label] = 1.

        return data, tlabel
            
    def __len__(self):
        return len(self.df)