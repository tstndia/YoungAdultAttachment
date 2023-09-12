import copy
import os.path as osp
import warnings
import mmcv
import numpy as np
import torch
import torchmetrics
import sys

from abc import ABCMeta, abstractmethod
from collections import OrderedDict, defaultdict
from torchmetrics.classification import MultilabelAccuracy
from torchmetrics.classification import MultilabelF1Score
from torchmetrics.classification import MultilabelPrecision
from torchmetrics.classification import MultilabelRecall
from torch import nn
from mmcv.utils import print_log
from torch.utils.data import Dataset

from ..core import (mean_average_precision, mean_class_accuracy,
                    mmit_mean_average_precision, top_k_accuracy)
from .pipelines import Compose
from .base import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class MultilabelVideoDataset(BaseDataset):
    """Video dataset for action recognition.

    The dataset loads raw videos and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    a sample video with the filepath and label, which are split with a
    whitespace. Example of a annotation file:

    .. code-block:: txt

        some/path/000.mp4 1
        some/path/001.mp4 1
        some/path/002.mp4 2
        some/path/003.mp4 2
        some/path/004.mp4 3
        some/path/005.mp4 3


    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking videos as input,
            it should be set to 0, since frames loaded from videos count
            from 0. Default: 0.
        **kwargs: Keyword arguments for ``BaseDataset``.
    """

    def __init__(self, ann_file, pipeline, start_index=0, **kwargs):
        super().__init__(ann_file, pipeline, start_index=start_index, **kwargs)

        self.loss_fn = nn.BCELoss()
        self.confusion_matrix = torchmetrics.ConfusionMatrix(8, multilabel=True)
        self.prec = MultilabelPrecision(num_labels=8)
        self.recall = MultilabelRecall(num_labels=8)
        self.f1_score = MultilabelF1Score(num_labels=8)
        self.accuracy = MultilabelAccuracy(num_labels=8)
    
    def load_annotations(self):
        """Load annotation file to get video information."""
        if self.ann_file.endswith('.json'):
            return self.load_json_annotations()

        video_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                line_split = line.strip().split()
                if self.multi_class:
                    assert self.num_classes is not None
                    filename, label = line_split[0], line_split[1:]
                    label = list(map(int, label))
                else:
                    filename, label = line_split
                    label = int(label)
                if self.data_prefix is not None:
                    filename = osp.join(self.data_prefix, filename)
                video_infos.append(dict(filename=filename, label=label))
                
        return video_infos

    def evaluate(self,
                 results,
                 metrics='top_k_accuracy',
                 metric_options=dict(top_k_accuracy=dict(topk=(1, 5))),
                 logger=None,
                 **deprecated_kwargs):
        """Perform evaluation for common datasets.

        Args:
            results (list): Output results.
            metrics (str | sequence[str]): Metrics to be performed.
                Defaults: 'top_k_accuracy'.
            metric_options (dict): Dict for metric options. Options are
                ``topk`` for ``top_k_accuracy``.
                Default: ``dict(top_k_accuracy=dict(topk=(1, 5)))``.
            logger (logging.Logger | None): Logger for recording.
                Default: None.
            deprecated_kwargs (dict): Used for containing deprecated arguments.
                See 'https://github.com/open-mmlab/mmaction2/pull/286'.

        Returns:
            dict: Evaluation results dict.
        """

        #labels = [ann['label'] for ann in self.video_infos]
        labels = []

        print(self.video_infos)
        for ann in self.video_infos:
            onehot = np.zeros(self.num_classes)
            onehot[ann['label']] = 1.
            labels.append(onehot)

        #print(results)
        #print(labels)

        results = torch.as_tensor(np.array(results), dtype=torch.float)
        gt_labels = torch.as_tensor(np.array(labels), dtype=torch.long)

        #results_sigmoid = results.sigmoid()

        np.set_printoptions(threshold=sys.maxsize)
        #print(f"preds: {results_sigmoid.numpy()}")
        #print(f"labels: {gt_labels.numpy()}")

        loss = self.loss_fn(results, gt_labels.float())
        cm = self.confusion_matrix(results, gt_labels)
        accuracy = self.accuracy(results, gt_labels)
        f1_score = self.f1_score(results, gt_labels)
        precision = self.prec(results, gt_labels)
        recall = self.recall(results, gt_labels)

        cm_mean = cm.float().mean(0)

        eval_results = OrderedDict()
        eval_results['test_loss'] = loss
        eval_results['accuracy'] = accuracy

        for i in range(cm.shape[0]):
            cmm = cm[i]
            eval_results[f'TN{i}'] = cmm[0,0]
            eval_results[f'FP{i}'] = cmm[0,1]
            eval_results[f'FN{i}'] = cmm[1,0]
            eval_results[f'TP{i}'] = cmm[1,1]

        eval_results['TN'] = cm_mean[0,0]
        eval_results['FP'] = cm_mean[0,1]
        eval_results['FN'] = cm_mean[1,0]
        eval_results['TP'] = cm_mean[1,1]

        eval_results['precision'] = precision
        eval_results['recall'] = recall
        eval_results['f1_score'] = f1_score
        
        return eval_results
