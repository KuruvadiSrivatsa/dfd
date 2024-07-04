import json
import glob
import os
import numpy as np
import cv2
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt
import face_recognition
from tqdm.autonotebook import tqdm

# Change the path accordingly
video_files = glob.glob('VideoCheck/deepfake-detection-challenge/test/fake/*.mp4')

frame_count = []
filtered_video_files = []

# Filter videos with less than 150 frames
for video_file in video_files:
    cap = cv2.VideoCapture(video_file)
    frame_count_current = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count_current >= 150:
        filtered_video_files.append(video_file)
        frame_count.append(frame_count_current)
    cap.release()

print("frames", frame_count)
print("Total number of videos:", len(frame_count))
print('Average frame per video:', np.mean(frame_count))

# Function to extract frames from a video
def frame_extract(path):
    vidObj = cv2.VideoCapture(path)
    success = True
    while success:
        success, image = vidObj.read()
        if success:
            yield image
    vidObj.release()

# Function to process frames and create videos with detected faces
def create_face_videos(path_list, out_dir):
    already_present_count = glob.glob(os.path.join(out_dir, '*.mp4'))
    print("No of videos already present", len(already_present_count))
    
    for path in tqdm(path_list):
        out_path = os.path.join(out_dir, os.path.basename(path))
        if os.path.exists(out_path):
            print("File Already exists:", out_path)
            continue

        frames = []
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (112, 112))
        
        for idx, frame in enumerate(frame_extract(path)):
            if idx > 150:
                break
            frames.append(frame)
            if len(frames) == 4:
                faces = face_recognition.batch_face_locations(frames)
                for i, face in enumerate(faces):
                    if len(face) != 0:
                        top, right, bottom, left = face[0]
                        try:
                            face_frame = frames[i][top:bottom, left:right, :]
                            resized_frame = cv2.resize(face_frame, (112, 112))
                            out.write(resized_frame)
                        except Exception as e:
                            print(f"Error processing frame {i} in video {path}: {e}")
                frames = []
        
        out.release()

create_face_videos(filtered_video_files, 'VideoCheck/preprocessed/fake')


