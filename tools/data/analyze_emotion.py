import argparse
import os
import os.path as osp
import pandas as pd
import torch
import cv2
import logging
import imutils
import numpy as np
from pathlib import Path
from deepface import DeepFace
from glob import glob
import tensorflow as tf
import multiprocessing
from concurrent.futures import wait, ALL_COMPLETED
from concurrent.futures.process import ProcessPoolExecutor

EMOTIONS = ['neutral', 'happy', 'sad', 'contempt', 'angry', 'disgust', 'surprise', 'fear']

def emotion_to_index(emotion):
    return EMOTIONS.index(emotion)
    
def list_to_file(lst, filename):
    with open(filename, 'w') as fp:
        for item in lst:
            fp.write("%s\n" % item)

def load_video(filename: str):
    """Loads a video from a file.

    Args:
        filename (str): filename of video

    Returns:
        A np.ndarray with dimensions (channels=3, frames, height, width). The
        values will be uint8's ranging from 0 to 255.

    Raises:
        FileNotFoundError: Could not find `filename`
        ValueError: An error occurred while reading the video
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(filename)

    try:
        capture = cv2.VideoCapture(filename)

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(capture.get(cv2.CAP_PROP_FPS))

        v = np.zeros((frame_count, frame_height, frame_width, 3), np.uint8) # (F, H, W, C)

        for count in range(frame_count):
            ret, frame = capture.read()
            
            if not ret:
                raise ValueError("Failed to load frame #{} of {}.".format(count, filename))

            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            v[count] = frame

            count += 1

        capture.release()

        assert v.size > 0

        return fps, v
    except Exception as e:
        raise e


def analyze_faces(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    videos = glob(os.path.join(input_dir, '*.mp4'))
    videos.sort()
    unanalyzed_videos = []

    for video in videos:
        filename = Path(video).stem
        out_filename = os.path.join(output_dir, filename, '.txt')

        if os.path.exists(out_filename):
            logging.info(f"File {filename} already analyzed. Skipping")
        else:
            unanalyzed_videos.append(video)

    print(f"Analyzed video: {len(videos) - len(unanalyzed_videos)}")
    print(f"Unanalyzed video: {len(unanalyzed_videos)}\n")
    print("Start analyzing ...")

    all_emotions = []

    '''
    with multiprocessing.Pool() as pool:
        items = [(video, output_dir, idx, len(videos)) 
            for idx, video in enumerate(videos)]

        for emotion in pool.map(analyze_face, items):
            all_emotions.append(emotion)

        pool.close
    '''
    items = [(video, output_dir, idx, len(videos)) 
            for idx, video in enumerate(videos)]
    
    for item in items:
        emotion = analyze_face(item)
        all_emotions.append(",".join(emotion))
    
    list_to_file(all_emotions, os.path.join(output_dir, 'all_emotion.txt'))

def analyze_face(task):
    video, output_dir, idx, total_video = task
    p_name = multiprocessing.current_process().name
    video_path = Path(video)
    filename = video_path.name
    out_filename = os.path.join(output_dir, video_path.stem + '.txt')
    frames = None

    try:
        logging.info(f"[{p_name}] Processing {idx + 1} of {total_video}: {filename}")
        fps, frames = load_video(video)
        logging.info(f"[{p_name}] Video shape: {frames.shape}, fps: {fps}")
    except Exception as e:
        logging.info(f"[{p_name}] Error load {filename} : {e}")

    all_emotions = [0 for _ in range(len(EMOTIONS))]
    emotions = []

    for fidx, frame in enumerate(frames):
        try:
            result = DeepFace.analyze(
                img_path = frame.squeeze(), actions = ['emotion']
            )

            emotions.append(f"Frame-{fidx} : {result['dominant_emotion']}")
            all_emotions[emotion_to_index(result['dominant_emotion'])] = 1
        except Exception as e:
            logging.info(f"[{p_name}] No face detected on frame: {fidx}. Skipping ==> {e}")

    if len(emotions) > 0:
        list_to_file(emotions, out_filename)

    all_emotions.insert(0, filename)
    
    return all_emotions

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze emotion face from videos')
    parser.add_argument("--input_dir", type=str, default="/workspace/AttachmentDewasa/data/exposure/", help="Input directory")
    parser.add_argument("--output_dir", type=str, default="/workspace/AttachmentDewasa/data/exposure_emotions/", help="Output directory")
    args = parser.parse_args()

    return args

def init_tf():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

if __name__ == '__main__':
    args = parse_args()
    init_tf()

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')

    analyze_faces(args.input_dir, args.output_dir)
