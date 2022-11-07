import argparse
import glob
import os
import os.path as osp
import pandas as pd
from pathlib import Path

def convert_csv_to_ann(csv_path):
    df = pd.read_csv(csv_path)
    emotions = ['neutral', 'happy', 'sad', 'contempt', 'anger', 'disgust', 'surprised', 'fear']
    class_emotions = []
    rows = []

    for index, row in df.iterrows():
        for idx, emotion in enumerate(emotions):
            if int(row[emotion]) == 1:
                class_emotions.append(idx + 1)

        items = " ".join(map(str, class_emotions))
        rows.append(f"{row['video_name']} {items}")
        class_emotions = []

    csv_path = Path(csv_path)
    with open(os.path.join(csv_path.parent, csv_path.stem, ".txt"), 'w') as f:
        for row in rows:
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
