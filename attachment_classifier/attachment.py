from tabnanny import verbose
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import numpy as np
import torch.nn.functional as F

from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.classification import MulticlassF1Score
from torchmetrics.classification import MulticlassPrecision
from torchmetrics.classification import MulticlassRecall
from torch.nn import functional as F

from einops import rearrange

class AttachmentClassifier(pl.LightningModule):
    def __init__(self, in_channels, num_classes = 3):
        super().__init__()
        self.save_hyperparameters()
        self.loss_fn = nn.CrossEntropyLoss()
        self.confusion_matrix = torchmetrics.ConfusionMatrix(num_classes)
        self.accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.prec = MulticlassPrecision(num_classes=num_classes)
        self.f1_score = MulticlassF1Score(num_classes=num_classes)
        self.recall = MulticlassRecall(num_classes=num_classes)
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(in_channels),
            nn.Linear(in_features=in_channels, out_features=8*in_channels, bias=True),
            nn.LeakyReLU(inplace=True),

            nn.Dropout(p=0.2),
            nn.BatchNorm1d(8*in_channels),
            nn.Linear(in_features=8*in_channels, out_features=4*in_channels, bias=True),
            nn.LeakyReLU(inplace=True),

            #nn.Dropout(p=0.2),
            #nn.BatchNorm1d(8*in_channels),
            #nn.Linear(in_features=8*in_channels, out_features=4*in_channels, bias=True),
            #nn.LeakyReLU(inplace=True),

            nn.Dropout(p=0.2),
            nn.BatchNorm1d(4*in_channels),
            nn.Linear(in_features=4*in_channels, out_features=2*in_channels, bias=True),
            nn.LeakyReLU(inplace=True),

            nn.BatchNorm1d(2*in_channels),
            nn.Linear(in_features=2*in_channels, out_features=num_classes)
        )

    def forward(self, x):
        x = self.classifier(x)

        return x

    def training_step(self, batch, batch_idx):
        predictions = self(batch[0])
        labels = batch[1]
        predictions_prob = predictions.softmax(dim=1)

        labels = torch.argmax(labels, dim=1)
        loss = self.loss_fn(predictions_prob, labels)
        acc = self.accuracy(predictions_prob, labels)

        self.log("loss", loss, on_epoch=True, on_step=True, prog_bar=True, logger=False)
        self.log("acc", acc, on_epoch=True, on_step=True, prog_bar=True, logger=False)

        self.logger.experiment.add_scalars('loss', {'train': loss}, global_step=self.current_epoch) 
        self.logger.experiment.add_scalars('acc', {'train': acc}, global_step=self.current_epoch) 

        return loss

    def validation_step(self, batch, batch_idx):
        predictions = self(batch[0])
        labels = batch[1]
        predictions_prob = predictions.softmax(dim=1)

        labels = torch.argmax(labels, dim=1)
        
        loss = self.loss_fn(predictions_prob, labels)
        acc = self.accuracy(predictions_prob, labels)

        self.log("val_loss", loss, on_epoch=True, on_step=True, prog_bar=True, logger=False)
        self.log("val_acc", acc, on_epoch=True, on_step=True, prog_bar=True, logger=False)

        self.logger.experiment.add_scalars('loss', {'val': loss}, global_step=self.current_epoch) 
        self.logger.experiment.add_scalars('acc', {'val': acc}, global_step=self.current_epoch) 
        
    def test_step(self, batch, batch_idx):
        predictions = self(batch[0])
        labels = batch[1]
        predictions_prob = predictions.softmax(dim=1)
        # print(predictions_prob)

        labels = torch.argmax(labels, dim=1)

        loss = self.loss_fn(predictions_prob, labels)
        cm = self.confusion_matrix(predictions_prob, labels.long())
        self.accuracy(predictions_prob, labels)
        self.f1_score(predictions_prob, labels)
        self.prec(predictions_prob, labels)
        self.recall(predictions_prob, labels)

        #cm_mean = cm.float().mean(0)
        
        #true negatives for class i in M(0,0)
        #false positives for class i in M(0,1)
        #false negatives for class i in M(1,0)
        #true positives for class i in M(1,1)

        self.log('test_loss', loss, on_epoch=True)
        self.log('accuracy', self.accuracy, on_epoch=True)

        self.log('TN', cm[0,0], on_epoch=True)
        self.log('FP', cm[0,1], on_epoch=True)
        self.log('FN', cm[1,0], on_epoch=True)
        self.log('TP', cm[1,1], on_epoch=True)

        self.log('precision', self.prec, on_epoch=True)
        self.log('recall', self.recall, on_epoch=True)
        self.log('f1_score', self.f1_score, on_epoch=True)

        total = torch.argmax(np.unique(predictions_prob.cpu().numpy(), return_counts=True))
        print(total)

    def predict_step(self, batch, batch_idx):
        return self.shared_step(batch, 'predict')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.0005)
        #optimizer = torch.optim.AdamW(self.parameters())
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, verbose=True, gamma=0.1)

        return [optimizer], [lr_scheduler]