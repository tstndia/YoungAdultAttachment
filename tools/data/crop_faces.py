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
import multiprocessing
from concurrent.futures import wait, ALL_COMPLETED
from concurrent.futures.process import ProcessPoolExecutor

def convert_and_trim_bb(image, rect):
	# extract the starting and ending (x, y)-coordinates of the
	# bounding box
	startX = rect.left()
	startY = rect.top()
	endX = rect.right()
	endY = rect.bottom()
	# ensure the bounding box coordinates fall within the spatial
	# dimensions of the image
	startX = max(0, startX)
	startY = max(0, startY)
	endX = min(endX, image.shape[1])
	endY = min(endY, image.shape[0])
	# compute the width and height of the bounding box
	w = endX - startX
	h = endY - startY
	# return our bounding box coordinates
	return (startX, startY, w, h)

def image_resize(image, width = None, height = None):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim)

    # return the resized image
    return resized

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

def save_video(name, video, fps, convert_to_bgr = True):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    data = cv2.VideoWriter(name, fourcc, float(fps), (video.shape[1], video.shape[2]))

    for frame in video:
        if convert_to_bgr:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        data.write(frame)

    data.release()

def crop_faces(input_dir, output_dir, detector, dim):
    videos = glob(os.path.join(input_dir, '*.mp4'))
    videos.sort()

    uncropped_videos = []

    for video in videos:
        filename = Path(video).name
        out_filename = os.path.join(output_dir, filename)

        if os.path.exists(out_filename):
            logging.info(f"File {filename} already cropped. Skipping")
        else:
            uncropped_videos.append(video)

    print(f"\nCropped video: {len(videos) - len(uncropped_videos)}")
    print(f"Uncropped video: {len(uncropped_videos)}\n")
    print("Start cropping ...")
    #pool = ProcessPoolExecutor()
    #pool.submit(lambda: None)
    
    with multiprocessing.Pool(processes=60) as pool:
        items = [(video, output_dir, detector, dim, idx, len(uncropped_videos)) 
            for idx, video in enumerate(uncropped_videos)]
        pool.imap(crop_face, items, chunksize = 20)
    #futures = [pool.submit(crop_face, item) for item in items]
    #wait(futures, return_when=ALL_COMPLETED)
    #pool.shutdown()
        pool.close

def crop_face(task):
    video, output_dir, detector, dim, idx, total_video = task
    p_name = multiprocessing.current_process().name
    filename = Path(video).name
    out_filename = os.path.join(output_dir, filename)

    frames = None

    try:
        logging.info(f"[{p_name}] Processing {idx + 1} of {total_video}: {filename}")
        fps, frames = load_video(video)
        logging.info(f"[{p_name}] Video shape: {frames.shape}, fps: {fps}")
    except Exception as e:
        logging.info(f"[{p_name}] Error load {filename} : {e}")

    faces = []

    for i in range(frames.shape[0]):
        try:
            frame = imutils.resize(frames[i,:,:,:].squeeze(), height=256)
            
            face = DeepFace.detectFace(img_path = frame, 
                target_size = (dim, dim),
                detector_backend = detector,
                align = False
            )

            faces.append((face * 255).astype(np.uint8))
        except Exception as e:
            logging.info(f"[{p_name}] No face detected on frame: {i}. Skipping ==> {e}")
            # pass

    if len(faces) > 0:
        cropped = np.stack(faces, axis=0)
        logging.info(f"[{p_name}] Finished. Cropped shape: {cropped.shape}. Total removed frames: {len(frames) - len(faces)}")
        save_video(out_filename, np.stack(faces, axis=0), fps)
        logging.info(f"[{p_name}] Saved into: {out_filename}")

def parse_args():
    parser = argparse.ArgumentParser(
        description='Crop face from videos')
    parser.add_argument("--input_dir", type=str, default="/workspace/AttachmentDewasa/data/exposure_full/", help="Input directory")
    parser.add_argument("--output_dir", type=str, default="/workspace/AttachmentDewasa/data/exposure/", help="Output directory")
    parser.add_argument("--detector", type=str, default='dlib', help="Detector backend")
    parser.add_argument("--dim", type=int, default=128, help="Output dimension")
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')

    crop_faces(args.input_dir, args.output_dir, args.detector, args.dim)
