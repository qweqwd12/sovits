from torch.utils.data import DataLoader
from vits.data_utils import DistributedBucketSampler
from vits.data_utils import TextAudioSpeakerCollate
from vits.data_utils import TextAudioSpeakerSet
from .data_utils import read_metadata_csv
import os
import csv
import torch
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import numpy as np


class TTSDataset(Dataset):
    def __init__(self, hp, meta_file):
        self.hp = hp
        self.data = []
        with open(meta_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='|')
            for row in reader:
                audio_path, text = row
                self.data.append((audio_path, text))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path, text = self.data[idx]
        audio, sr = sf.read(audio_path)
        audio = torch.FloatTensor(audio)
        return audio, text
    
def create_dataloader_train(hps, n_gpus, rank, meta_file):
    collate_fn = TextAudioSpeakerCollate()
    #添加读取csv
    dataset = TTSDataset(hps,meta_file)

    train_dataset = TextAudioSpeakerSet(hps.data.training_files, hps.data)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        [150, 300, 450],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler)
    return train_loader


def create_dataloader_eval(hps):
    collate_fn = TextAudioSpeakerCollate()

    metadata = read_metadata_csv('/root/autodl-tmp/so-vits-svc-5.0/dataset_raw/losl/metadata.csv')
    
    eval_dataset = TextAudioSpeakerSet(hps.data.validation_files, hps.data, metadata=metadata)
    eval_loader = DataLoader(
        eval_dataset,
        num_workers=2,
        shuffle=False,
        batch_size=hps.train.batch_size,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn)
    return eval_loader

