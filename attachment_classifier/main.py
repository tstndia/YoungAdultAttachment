import argparse
from unittest.mock import patch
import pytorch_lightning as pl
from attachment import AttachmentClassifier
from attachment_datamodule import AttachmentDataModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="train", help="Train or test")
parser.add_argument("--data_dir", type=str, default="data/attachments", help="Path ke datasets")
parser.add_argument("--modality", type=str, default=None, help="Path ke datasets")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
parser.add_argument("--ckpt_path", type=str, default=None, help="Checkpoint path")
parser.add_argument("--max_epochs", type=int, default=200, help="Max epochs")
parser.add_argument("--num_workers", type=int, default=4, help="num_workers")
parser.add_argument("--accelerator", type=str, default='gpu', help="Accelerator")
parser.add_argument("--logs_dir", type=str, default='lightning_logs', help="Log dir")
parser.add_argument("--log", action='store_true', help="log")

params = parser.parse_args()

if __name__ == '__main__':
    mode = params.mode
    data_dir = params.data_dir
    modality = params.modality
    batch_size = params.batch_size
    ckpt_path = params.ckpt_path
    max_epochs = params.max_epochs
    num_workers = params.num_workers
    accelerator = params.accelerator
    logs_dir = params.logs_dir
    log = params.log

    experiment_name = Path(data_dir).name + "_" + modality
    logger = TensorBoardLogger(save_dir=logs_dir, name=experiment_name)

    data_module = AttachmentDataModule(data_dir=data_dir, 
                        batch_size=batch_size, 
                        num_workers=num_workers,
                        modality=modality)

    in_channels = 8 * 14 + 36
    # single modality ablation (exp:exposure, respv: responsevideo, sra: speechresponseaudio)
    # if modality == 'exp-respv-sra':
    #     in_channels = 3 * 8 * 14
    # elif modality == 'exp-respv-quest' or modality == 'exp-sra-quest' or modality == 'respv-sra-quest':
    #     in_channels = 2 * 8 * 14 + 36
    # elif modality == 'exp-quest' or modality == 'respv-quest' or modality == 'sra-quest':
    #     in_channels = 8 * 14 + 36
    # elif modality == 'exp-respv' or modality == 'exp-sra' or modality == 'respv-sra':
    #     in_channels = 2 * 8 * 14
    # elif modality == 'exposure' or modality == 'video_response' or modality == 'audio_response':
    #     in_channels = 8 * 14
    # else:
    #     in_channels = 36


    attachmentClassifier = AttachmentClassifier(
            in_channels = in_channels, 
            num_classes = 3)

    trainer = pl.Trainer(
                accelerator=accelerator, 
                max_epochs=max_epochs, 
                num_sanity_val_steps=1, 
                auto_scale_batch_size=True, 
                enable_model_summary=True,
                logger=logger,
                callbacks=[EarlyStopping(monitor="val_loss")])

    if mode == 'train':
        trainer.fit(model=attachmentClassifier, datamodule=data_module, ckpt_path=ckpt_path)

    if mode == 'validate':
        if not log:
            trainer.logger = False

        trainer.validate(model=attachmentClassifier, datamodule=data_module, ckpt_path=ckpt_path)

    if mode == 'test':
        if not log:
            trainer.logger = False

        trainer.test(model=attachmentClassifier, datamodule=data_module, ckpt_path=ckpt_path)

    if mode == 'predict':
        if not log:
            trainer.logger = False

        predicts = trainer.predict(model=attachmentClassifier, datamodule=data_module, ckpt_path=ckpt_path)

        for predict in predicts:
            print(predict)
            print('\n')
