import argparse
import os
import os.path as osp
import warnings
import re

import mmcv
import torch
import numpy as np
import pandas as pd
import csv
from pathlib import Path
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.fileio.io import file_handlers
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.runner.fp16_utils import wrap_fp16_model
from sklearn.model_selection import train_test_split

from mmaction.datasets import build_dataloader, build_dataset
from mmaction.models import build_model
from mmaction.utils import register_module_hooks

# TODO import test functions from mmcv and delete them from mmaction2
try:
    from mmcv.engine import multi_gpu_test#, single_gpu_test
except (ImportError, ModuleNotFoundError):
    warnings.warn(
        'DeprecationWarning: single_gpu_test, multi_gpu_test, '
        'collect_results_cpu, collect_results_gpu from mmaction2 will be '
        'deprecated. Please install mmcv through master branch.')
    from mmaction.apis import multi_gpu_test, single_gpu_test


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMAction2 test (and eval) a model')
    parser.add_argument('config_exposure', help='test config file path')
    parser.add_argument('config_response', help='test config file path')
    parser.add_argument('config_stimuli', help='test config file path')
    parser.add_argument('checkpoint_exposure', help='checkpoint file')
    parser.add_argument('checkpoint_response', help='checkpoint file')
    parser.add_argument('checkpoint_stimuli', help='checkpoint file')
    parser.add_argument('csv_path', help='test config file path')
    parser.add_argument(
        '--out',
        default=None,
        help='output result file in pkl/yaml/json format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g.,'
        ' "top_k_accuracy", "mean_class_accuracy" for video dataset')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        default={},
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        default={},
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--average-clips',
        choices=['score', 'prob', None],
        default=None,
        help='average type when averaging test clips')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--onnx',
        action='store_true',
        help='Whether to test with onnx model or not')
    parser.add_argument(
        '--tensorrt',
        action='store_true',
        help='Whether to test with TensorRT engine or not')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def turn_off_pretrained(cfg):
    # recursively find all pretrained in the model config,
    # and set them None to avoid redundant pretrain steps for testing
    if 'pretrained' in cfg:
        cfg.pretrained = None

    # recursively turn off pretrained value
    for sub_cfg in cfg.values():
        if isinstance(sub_cfg, dict):
            turn_off_pretrained(sub_cfg)

def single_gpu_test(model, data_loader):  # noqa: F811
    """Test model with a single gpu.

    This method tests model with a single gpu and
    displays test progress bar.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    filenames = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        with torch.no_grad():
            result = model(return_loss=False, **data)
        results.extend(result)

        if 'filename' in data['img_metas'].data[0][0]:
            filenames.append(data['img_metas'].data[0][0]['filename'])
        else:
            filenames.append(data['img_metas'].data[0][0]['audio_path'])

        # use the first key as main key to calculate the batch size
        batch_size = len(next(iter(data.values())))
        for _ in range(batch_size):
            prog_bar.update()
    return results, filenames
    
def inference_pytorch(args, cfg, distributed, data_loader):
    """Get predictions by pytorch models."""
    if args.average_clips is not None:
        # You can set average_clips during testing, it will override the
        # original setting
        if cfg.model.get('test_cfg') is None and cfg.get('test_cfg') is None:
            cfg.model.setdefault('test_cfg',
                                 dict(average_clips=args.average_clips))
        else:
            if cfg.model.get('test_cfg') is not None:
                cfg.model.test_cfg.average_clips = args.average_clips
            else:
                cfg.test_cfg.average_clips = args.average_clips

        cfg.model.test_cfg.average_clips = 'score'

    # remove redundant pretrain steps for testing
    turn_off_pretrained(cfg.model)

    # build the model and load checkpoint
    model = build_model(
        cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))

    if len(cfg.module_hooks) > 0:
        register_module_hooks(model, cfg.module_hooks)

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    return outputs


def inference_tensorrt(ckpt_path, distributed, data_loader, batch_size):
    """Get predictions by TensorRT engine.

    For now, multi-gpu mode and dynamic tensor shape are not supported.
    """
    assert not distributed, \
        'TensorRT engine inference only supports single gpu mode.'
    import tensorrt as trt
    from mmcv.tensorrt.tensorrt_utils import (torch_dtype_from_trt,
                                              torch_device_from_trt)

    # load engine
    with trt.Logger() as logger, trt.Runtime(logger) as runtime:
        with open(ckpt_path, mode='rb') as f:
            engine_bytes = f.read()
        engine = runtime.deserialize_cuda_engine(engine_bytes)

    # For now, only support fixed input tensor
    cur_batch_size = engine.get_binding_shape(0)[0]
    assert batch_size == cur_batch_size, \
        ('Dataset and TensorRT model should share the same batch size, '
         f'but get {batch_size} and {cur_batch_size}')

    context = engine.create_execution_context()

    # get output tensor
    dtype = torch_dtype_from_trt(engine.get_binding_dtype(1))
    shape = tuple(context.get_binding_shape(1))
    device = torch_device_from_trt(engine.get_location(1))
    output = torch.empty(
        size=shape, dtype=dtype, device=device, requires_grad=False)

    # get predictions
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        bindings = [
            data['imgs'].contiguous().data_ptr(),
            output.contiguous().data_ptr()
        ]
        context.execute_async_v2(bindings,
                                 torch.cuda.current_stream().cuda_stream)
        results.extend(output.cpu().numpy())
        batch_size = len(next(iter(data.values())))
        for _ in range(batch_size):
            prog_bar.update()
    return results


def inference_onnx(ckpt_path, distributed, data_loader, batch_size):
    """Get predictions by ONNX.

    For now, multi-gpu mode and dynamic tensor shape are not supported.
    """
    assert not distributed, 'ONNX inference only supports single gpu mode.'

    import onnx
    import onnxruntime as rt

    # get input tensor name
    onnx_model = onnx.load(ckpt_path)
    input_all = [node.name for node in onnx_model.graph.input]
    input_initializer = [node.name for node in onnx_model.graph.initializer]
    net_feed_input = list(set(input_all) - set(input_initializer))
    assert len(net_feed_input) == 1

    # For now, only support fixed tensor shape
    input_tensor = None
    for tensor in onnx_model.graph.input:
        if tensor.name == net_feed_input[0]:
            input_tensor = tensor
            break
    cur_batch_size = input_tensor.type.tensor_type.shape.dim[0].dim_value
    assert batch_size == cur_batch_size, \
        ('Dataset and ONNX model should share the same batch size, '
         f'but get {batch_size} and {cur_batch_size}')

    # get predictions
    sess = rt.InferenceSession(ckpt_path)
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        imgs = data['imgs'].cpu().numpy()
        onnx_result = sess.run(None, {net_feed_input[0]: imgs})[0]
        results.extend(onnx_result)
        batch_size = len(next(iter(data.values())))
        for _ in range(batch_size):
            prog_bar.update()
    return results

def infer(cfg, dataset, distributed, args):
    # build the dataloader
    #dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    dataloader_setting = dict(
        videos_per_gpu=1, #cfg.data.get('videos_per_gpu', 1),
        workers_per_gpu=1, #cfg.data.get('workers_per_gpu', 1),
        dist=distributed,
        shuffle=False)
    dataloader_setting = dict(dataloader_setting,
                              **cfg.data.get('test_dataloader', {}))
    data_loader = build_dataloader(dataset, **dataloader_setting)

    #a = iter(data_loader).next()

    #print(a['imgs'].shape)
    #print(a['label'].shape)
    #print(a['img_metas'].data[0][0]['filename'])

    if args.tensorrt:
        outputs = inference_tensorrt(args.checkpoint, distributed, data_loader,
                                     dataloader_setting['videos_per_gpu'])
    elif args.onnx:
        outputs = inference_onnx(args.checkpoint, distributed, data_loader,
                                 dataloader_setting['videos_per_gpu'])
    else:
        outputs = inference_pytorch(args, cfg, distributed, data_loader)

    return outputs

def load_cfg(config, args):
    cfg = Config.fromfile(config)

    cfg.merge_from_dict(args.cfg_options)

    # Load output_config from cfg
    output_config = cfg.get('output_config', {})
    if args.out:
        # Overwrite output_config from args.out
        output_config = Config._merge_a_into_b(
            dict(out=args.out), output_config)

    # Load eval_config from cfg
    eval_config = cfg.get('eval_config', {})
    if args.eval:
        # Overwrite eval_config from args.eval
        eval_config = Config._merge_a_into_b(
            dict(metrics=args.eval), eval_config)
    if args.eval_options:
        # Add options from args.eval_options
        eval_config = Config._merge_a_into_b(args.eval_options, eval_config)

    assert output_config or eval_config, \
        ('Please specify at least one operation (save or eval the '
         'results) with the argument "--out" or "--eval"')

    dataset_type = cfg.data.test.type
    if output_config.get('out', None):
        if 'output_format' in output_config:
            # ugly workround to make recognition and localization the same
            warnings.warn(
                'Skip checking `output_format` in localization task.')
        else:
            out = output_config['out']
            # make sure the dirname of the output path exists
            mmcv.mkdir_or_exist(osp.dirname(out))
            _, suffix = osp.splitext(out)
            if dataset_type == 'AVADataset':
                assert suffix[1:] == 'csv', ('For AVADataset, the format of '
                                             'the output file should be csv')
            else:
                assert suffix[1:] in file_handlers, (
                    'The format of the output '
                    'file should be json, pickle or yaml')

    # set cudnn benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # The flag is used to register module's hooks
    cfg.setdefault('module_hooks', [])

    return cfg

def main():
    args = parse_args()

    attachment_path = os.path.join('data', 'attachments')
    attachment_train = os.path.join('data', 'attachments', 'train')
    attachment_test = os.path.join('data', 'attachments', 'test')

    if not os.path.exists(attachment_path):
        os.makedirs(attachment_path)

    if not os.path.exists(attachment_train):
        os.makedirs(attachment_train)

    if not os.path.exists(attachment_test):
        os.makedirs(attachment_test)
    
    cfg_exposure = load_cfg(args.config_exposure, args)
    cfg_video = load_cfg(args.config_response, args)
    cfg_audio = load_cfg(args.config_stimuli, args)

    df = pd.read_csv(args.csv_path)
    data_df = df[['name']]
    label_df = df[['attachment_type']]

    X_train, X_test, y_train, y_test = train_test_split(data_df.values, label_df.values, 
        test_size=0.2, random_state=42)
    
    configs = [cfg_exposure, cfg_video, cfg_audio]
    modalities = ['exposure', 'response', 'stimuli']
    resps = dict()

    for cfg, modality in zip(configs, modalities):
        train_dataset = build_dataset(cfg.data.train, dict(test_mode=True))
        test_dataset = build_dataset(cfg.data.test, dict(test_mode=True))

        datasets = [train_dataset, test_dataset]
        args.checkpoint = getattr(args, f"checkpoint_{modality}")

        for dataset in datasets:
            outputs, filenames = infer(cfg, dataset, False, args)

            for output, filename in zip(outputs, filenames):
                fn = Path(filename).stem
                resp, stimuli = re.findall("^([a-zA-Z]+)_([a-zA-Z0-9]+)", fn)[0]

                # initialize data
                if resp not in resps:
                    data = dict()

                    for modality in modalities:
                        for i in range(0, 8):
                            data[f"{modality}{i+1}"] = list(np.zeros(8))

                    resps[resp] = data

                resps[resp][stimuli] = output

    splits = dict({'train': (X_train, y_train), 'test': (X_test, y_test)})

    for split in splits:
        #X = splits[split][0]
        #y = splits[split][1]

        for item in splits[split]:
            names, labels = item[0], item[1]

            for name, label in zip(names, labels):
                resp = resps[name]
                stimulis = []

                for stimuli in sorted(resp):
                    stimulis.append(stimuli)

                nps = np.append(stimulis)
                np.save(os.path.join(attachment_path, split, f"{name}.npy"), nps)

        with open(os.path.join(attachment_path, f"{split}.csv"), 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['filename', 'label'])

            for item in splits[split]:
                names, labels = item[0], item[1]
                
                for name, label in zip(names, labels):
                    writer.writerow([f"{name}.npy", label])
        
if __name__ == '__main__':
    main()
