import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import opensmile
from glob import glob
from pathlib import Path
import noisereduce as nr
import matplotlib.pyplot as plt

lenmin = 5168


def load_audio(path):
    y, sr = librosa.load(path)
    reduced_noise = nr.reduce_noise(y=y, sr=sr)
    mel = librosa.feature.mfcc(y=reduced_noise, sr=sr, n_mels=20)
    return mel


def prepare_audios(df, root_dir):
    num_samples = len(df)
    audio_paths = df["audio_name"].values.tolist()
    labels = df[["neutral", "happy", "sad", "contempt", "anger", "disgust", "surprised", "fear"]].values

    mfccs = []
    fls = []

    for idx, path in enumerate(audio_paths):
        mel = load_audio(os.path.join(root_dir, path))
        print(mel.shape[0])
        print(mel.shape[1])
        if mel.shape[1] < lenmin:
            mel = np.hstack((mel, np.zeros((mel.shape[0], lenmin - mel.shape[1]))))
        mel = mel[:, :lenmin]
        mfccs.append(mel.T)
        print(path)
    mfcc = np.array(mfccs)
    return mfcc
#         mfccs.append(mel)
#         fls.append(fl)
#     return mfccs, fls

path = "/workspace/AttachmentDewasa/data/response-audio/"
response_audio = pd.read_csv("/workspace/AttachmentDewasa/data/response-audio/response_audio.csv")
mfcc = prepare_audios(response_audio, path)
print(mfcc.shape)