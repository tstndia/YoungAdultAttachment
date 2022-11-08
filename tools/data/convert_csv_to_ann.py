import argparse
import glob
import os
import os.path as osp
import pandas as pd
from pathlib import Path
from skmultilearn.model_selection import iterative_train_test_split

def convert_csv_to_ann(csv_path):
    df = pd.read_csv(csv_path)
    data_df = df[["video_name"]]
    label_df = df[["neutral", "happy", "sad", "contempt", "anger", "disgust", "surprised", "fear"]]

    data_train, label_train, data_test, label_test = iterative_train_test_split(
        data_df[["video_name"]].values,
        label_df[["neutral", "happy", "sad", "contempt", "anger", "disgust", "surprised", "fear"]].values,
        test_size = 0.2
    )

    print(data_train.shape)
    print(label_train.shape)

    emotions = ['neutral', 'happy', 'sad', 'contempt', 'anger', 'disgust', 'surprised', 'fear']
    class_emotions = []
    train_rows = []
    val_rows = []

    for index, video in enumerate(data_train):
        for idx, emotion in enumerate(emotions):
            if int(label_train[emotion]) == 1:
                class_emotions.append(idx + 1)

        items = " ".join(map(str, class_emotions))
        train_rows.append(f"{video['video_name']} {items}")
        class_emotions = []

    for index, video in enumerate(data_test):
        for idx, emotion in enumerate(emotions):
            if int(label_test[emotion]) == 1:
                class_emotions.append(idx + 1)

        items = " ".join(map(str, class_emotions))
        val_rows.append(f"{video['video_name']} {items}")
        class_emotions = []

    csv_path = Path(csv_path)

    with open(os.path.join(csv_path.parent, f"{csv_path.stem}_train.txt"), 'w') as f:
        for row in train_rows:
            f.write(f"{row}\n")
    
    with open(os.path.join(csv_path.parent, f"{csv_path.stem}_val.txt"), 'w') as f:
        for row in val_rows:
            f.write(f"{row}\n")

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert csv to annotation file')
    parser.add_argument('--csv_path', type=str, help='source video directory')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if not osp.exists(args.csv_path):
        print(f"File {args.csv_path} not exist")
        exit()
    
    convert_csv_to_ann(args.csv_path)
