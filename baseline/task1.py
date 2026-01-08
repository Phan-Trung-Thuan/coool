import cv2
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import *
from moviepy import VideoFileClip
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression

def scale(x):
    x = abs(x)
    if x.mean() == 0:
        return np.array([])
    return x/x.mean()


df_final = pd.DataFrame()
annotations = pickle.load(open("/kaggle/input/coool-dataset/annotations_public.pkl", 'rb'))

ids = []
Driver_State_Changed = []
video_Driver_State_Changed = {}

chunk = 16
for video in tqdm(sorted(annotations.keys())):
    a = {}
    speed = {}
    traffic_scene_frames = {}
    traffic_scene_centroids = {}
    challenge_object_frames = {}
    challenge_object_centroids = {}
    num_frames = len(annotations[video].keys())
    for frame in sorted(annotations[video].keys()):
        for i in range(len(annotations[video][frame]['traffic_scene'])):
            if annotations[video][frame]['traffic_scene'][i]['track_id'] in traffic_scene_centroids:
                x1, y1, x2, y2 = annotations[video][frame]['traffic_scene'][i]['bbox']
                traffic_scene_frames[annotations[video][frame]['traffic_scene'][i]['track_id']].append(frame)
                traffic_scene_centroids[annotations[video][frame]['traffic_scene'][i]['track_id']].append([x1+(abs(x2-x1)/2),y1+(abs(y2-y1)/2)])
            else:
                x1, y1, x2, y2 = annotations[video][frame]['traffic_scene'][i]['bbox']
                traffic_scene_frames[annotations[video][frame]['traffic_scene'][i]['track_id']] = [frame]
                traffic_scene_centroids[annotations[video][frame]['traffic_scene'][i]['track_id']] = [[x1+(abs(x2-x1)/2),y1+(abs(y2-y1)/2)]]
        for i in range(len(annotations[video][frame]['challenge_object'])):
            if annotations[video][frame]['challenge_object'][i]['track_id'] in challenge_object_centroids:
                x1, y1, x2, y2 = annotations[video][frame]['challenge_object'][i]['bbox']
                challenge_object_frames[annotations[video][frame]['challenge_object'][i]['track_id']].append(frame)
                challenge_object_centroids[annotations[video][frame]['challenge_object'][i]['track_id']].append([x1+(abs(x2-x1)/2),y1+(abs(y2-y1)/2)])
            else:
                x1, y1, x2, y2 = annotations[video][frame]['challenge_object'][i]['bbox']
                challenge_object_frames[annotations[video][frame]['challenge_object'][i]['track_id']] = [frame]
                challenge_object_centroids[annotations[video][frame]['challenge_object'][i]['track_id']] = [[x1+(abs(x2-x1)/2),y1+(abs(y2-y1)/2)]]

        for k in traffic_scene_frames:
            if frame in traffic_scene_frames[k]:
                if len(traffic_scene_frames[k]) > 1:
                    y = np.linalg.norm(np.array(traffic_scene_centroids[k][1:])-np.array(traffic_scene_centroids[k][:-1]), axis=1)
                    x = np.array(traffic_scene_frames[k][1:]).reshape(-1, 1)-traffic_scene_frames[k][0]
                    speed_model = LinearRegression().fit(x, y)
                    if k in speed:
                        speed[k].append(speed_model.coef_)
                        if len(speed[k]) > chunk:
                            if k in a:
                                a[k].append([frame, float(LinearRegression().fit(np.arange(chunk).reshape(-1, 1), y[-chunk:]).coef_[0])])
                            else:
                                a[k] = [[frame, float(LinearRegression().fit(np.arange(chunk).reshape(-1, 1), y[-chunk:]).coef_[0])]]
                    else:
                        speed[k] = [speed_model.coef_]

        for k in challenge_object_frames:
            if frame in challenge_object_frames[k]:
                if len(challenge_object_frames[k]) > 1:
                    y = np.linalg.norm(np.array(challenge_object_centroids[k][1:])-np.array(challenge_object_centroids[k][:-1]), axis=1)
                    x = np.array(challenge_object_frames[k][1:]).reshape(-1, 1)-challenge_object_frames[k][0]
                    speed_model = LinearRegression().fit(x, y)
                    if k in speed:
                        speed[k].append(speed_model.coef_)
                        if len(speed[k]) > chunk:
                            if k in a:
                                a[k].append([frame, float(LinearRegression().fit(np.arange(chunk).reshape(-1, 1), y[-chunk:]).coef_[0])])
                            else:
                                a[k] = [[frame, float(LinearRegression().fit(np.arange(chunk).reshape(-1, 1), y[-chunk:]).coef_[0])]]
                    else:
                        speed[k] = [speed_model.coef_]
    speed_detect = {}
    for k, v in a.items():
        x = np.array(v)
        for d in x[:, 0][find_peaks(scale(x[:,1]),height=3.3, distance=40)[0]]:
            d = int(d)
            if d > chunk//2:
                if d in speed_detect:
                    speed_detect[d] += 1
                else:
                    speed_detect[d] = 1
                    
    sound_detect = {}
    try:
        clip = VideoFileClip(f"/kaggle/input/COOOL-videos/{video}.mp4")
    except:
        continue
        
    signal = scale(np.abs(list(clip.audio.iter_frames(fps=60)))[:, 0])
    peaks, _ = find_peaks(signal, height=3.3, distance=40)
    for peak in peaks:
        d = round(num_frames*peak/len(signal))
        if d > chunk:
            if d in sound_detect:
                sound_detect[d] += 1
            else:
                sound_detect[d] = 1

    signal = scale(np.abs(list(clip.audio.iter_frames(fps=60)))[:, 1])
    peaks, _ = find_peaks(signal, height=3.3, distance=40)
    for peak in peaks:
        d = round(num_frames*peak/len(signal))
        if d > chunk:
            if d in sound_detect:
                sound_detect[d] += 1
            else:
                sound_detect[d] = 1

    signal = scale(np.mean(np.abs(list(clip.audio.iter_frames(fps=60))), axis=1))
    peaks, _ = find_peaks(signal, height=3.3, distance=40)
    for peak in peaks:
        d = round(num_frames*peak/len(signal))
        if d > chunk:
            if d in sound_detect:
                sound_detect[d] += 1
            else:
                sound_detect[d] = 1

    ids += [f"{video}_{frame}" for frame in range(num_frames)]
    if len(sound_detect) == 0 and len(speed_detect) == 0:
        Driver_State_Changed += [False]*int(0.25*num_frames) + [True]*(num_frames-int(0.25*num_frames))
        video_Driver_State_Changed[video] = ["random", int(0.25*num_frames)]
    elif len(sound_detect) == 0:
        Driver_State_Changed += [frame >= min(speed_detect) for frame in range(num_frames)]
        video_Driver_State_Changed[video] = ["Speed", min(speed_detect)]
    elif len(speed_detect) == 0:
        Driver_State_Changed += [frame >= np.average(list(sound_detect.keys()), weights=list(sound_detect.values())) for frame in range(num_frames)]
        video_Driver_State_Changed[video] = ["Sound", np.average(list(sound_detect.keys()), weights=list(sound_detect.values()))]
    else:
        Driver_State_Changed += [frame >= min(speed_detect) for frame in range(num_frames)]
        video_Driver_State_Changed[video] = ["Speed+Sound", min(speed_detect), np.average(list(sound_detect.keys()), weights=list(sound_detect.values()))]

with open('video_Driver_State_Changed.pkl', 'wb') as handle:
    pickle.dump(video_Driver_State_Changed, handle, protocol=pickle.HIGHEST_PROTOCOL)