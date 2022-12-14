import pytorch_lightning as pl
import os
import pandas as pd
import cv2
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from attachment_dataset import AttachmentDataset

class AttachmentDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "data/attachments", 
            batch_size: int = 32, 
            num_workers: int = 8,
            num_classes: int = 3):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = num_classes
            
    def setup(self, stage = None):
        print(f'setup: {self.data_dir}')

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_set = AttachmentDataset(root=os.path.join(self.data_dir, 'train'), 
                csv_file=os.path.join(self.data_dir, 'train.csv'), 
                num_classes=self.num_classes)
            
            self.val_set = AttachmentDataset(root=os.path.join(self.data_dir, 'test'), 
                csv_file=os.path.join(self.data_dir, 'test.csv'), 
                num_classes=self.num_classes)

        if stage == "validate" or stage is None:
            self.val_set = AttachmentDataset(root=os.path.join(self.data_dir, 'test'), 
                csv_file=os.path.join(self.data_dir, 'test.csv'), 
                num_classes=self.num_classes)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.val_set = AttachmentDataset(root=os.path.join(self.data_dir, 'test'), 
                csv_file=os.path.join(self.data_dir, 'test.csv'), 
                num_classes=self.num_classes)

        if stage == "predict" or stage is None:
            self.val_set = AttachmentDataset(root=os.path.join(self.data_dir, 'test'), 
                csv_file=os.path.join(self.data_dir, 'test.csv'), 
                num_classes=self.num_classes)

    def train_dataloader(self):
        return DataLoader(self.train_set, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=True,
            drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.val_set, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            drop_last=True)

    def predict_dataloader(self):
        return DataLoader(self.val_set, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            drop_last=True)
