import os.path as osp

import torch
import numpy as np
from .base import BaseDataset
from .builder import DATASETS
from collections import OrderedDict, defaultdict
import torchmetrics
import sys
from torchmetrics.classification import MultilabelAccuracy
from torchmetrics.classification import MultilabelF1Score
from torchmetrics.classification import MultilabelPrecision
from torchmetrics.classification import MultilabelRecall
from torch import nn

@DATASETS.register_module()
class MultilabelAudioFeatureDataset(BaseDataset):
    """Audio feature dataset for video recognition. Reads the features
    extracted off-line. Annotation file can be that of the rawframe dataset,
    or:

    .. code-block:: txt

        some/directory-1.npy 163 1
        some/directory-2.npy 122 1
        some/directory-3.npy 258 2
        some/directory-4.npy 234 2
        some/directory-5.npy 295 3
        some/directory-6.npy 121 3

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        suffix (str): The suffix of the audio feature file. Default: '.npy'.
        kwargs (dict): Other keyword args for `BaseDataset`.
    """

    def __init__(self, ann_file, pipeline, suffix='.npy', **kwargs):
        self.suffix = suffix
        super().__init__(ann_file, pipeline, modality='Audio', **kwargs)

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
                video_info = {}
                idx = 0
                filename = line_split[idx]
                if self.data_prefix is not None:
                    if not filename.endswith(self.suffix):
                        filename = osp.join(self.data_prefix,
                                            filename) + self.suffix
                    else:
                        filename = osp.join(self.data_prefix, filename)
                video_info['audio_path'] = filename
                idx += 1
                # idx for total_frames
                video_info['total_frames'] = int(line_split[idx])
                idx += 1
                # idx for label[s]
                label = [int(x) for x in line_split[idx:]]
                assert label, f'missing label in line: {line}'
                if self.multi_class:
                    assert self.num_classes is not None
                    onehot = torch.zeros(self.num_classes)
                    onehot[label] = 1.0
                    video_info['label'] = onehot
                else:
                    assert len(label) == 1
                    video_info['label'] = label[0]
                video_infos.append(video_info)

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

        labels = [ann['label'].numpy() for ann in self.video_infos]
        #labels = []

        #for ann in self.video_infos:
        #    onehot = np.zeros(self.num_classes)
        #    onehot[ann['label']] = 1.
        #    labels.append(onehot)

        #print(results)
        #print(labels)

        results = torch.as_tensor(np.array(results), dtype=torch.float)
        gt_labels = torch.as_tensor(np.array(labels), dtype=torch.long)
        #gt_labels = labels.long()
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

        eval_results['TN'] = cm_mean[0,0]
        eval_results['FP'] = cm_mean[0,1]
        eval_results['FN'] = cm_mean[1,0]
        eval_results['TP'] = cm_mean[1,1]

        eval_results['precision'] = precision
        eval_results['recall'] = recall
        eval_results['f1_score'] = f1_score
        
        return eval_results
