import cv2
import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import *
from moviepy import VideoFileClip
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
from transformers import AutoImageProcessor, AutoModelForImageClassification

def scale(x):
    x = abs(x)
    if x.mean() == 0:
        return np.array([])
    return x/x.mean()

def is_point_in_quadrilateral(x, y, points):
    def cross_product(x1, y1, x2, y2, x, y):
        return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
    
    P1, P2, P3, P4 = points  # bottom_left, top_left, bottom_right, top_right
    C1 = cross_product(P1[0], P1[1], P2[0], P2[1], x, y)
    C2 = cross_product(P2[0], P2[1], P4[0], P4[1], x, y)
    C3 = cross_product(P4[0], P4[1], P3[0], P3[1], x, y)
    C4 = cross_product(P3[0], P3[1], P1[0], P1[1], x, y)

    # Check if all cross products have the same sign
    return (C1 >= 0 and C2 >= 0 and C3 >= 0 and C4 >= 0) or (C1 <= 0 and C2 <= 0 and C3 <= 0 and C4 <= 0)


def imagenet_map(x):
    if x in [407, 408, 468, 569, 609, 627, 654, 656, 675, 734, 817, 867, 874]:
        return "car"
    return model.config.id2label[x]

ids = []
df_final = pd.DataFrame()
annotations = pickle.load(open("/kaggle/input/coool-dataset/annotations_public.pkl", 'rb'))
processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-base-patch4-window16-256")
model = AutoModelForImageClassification.from_pretrained("microsoft/swinv2-base-patch4-window16-256").to("cuda:1", dtype=torch.float16)

processor2 = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model2 = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224").to("cuda:1", dtype=torch.float16)

Hazard_Track_all = [[] for i in range(23)]
video_track_id = {}
video_first_hazard = {}
weight = np.array([0.054, 1.0, 0.0012, 0.838, 0.935, 0.0484, 0.707818, -1.3])

for video in tqdm(sorted(list(annotations.keys()))):
    try:
        video_stream = cv2.VideoCapture(f"/kaggle/input/coool-dataset/COOOL-videos/{video}.mp4")
        
        challenge_object_frames = {}
        challenge_object_labels = {}
        challenge_object_labels2 = {}
        challenge_object_centroids = {}
        challenge_object_minx = {}
        challenge_object_maxx = {}
        num_frames = len(annotations[video].keys())
        
        all_centroids = {}
        count_frame = {}
        for frame in sorted(annotations[video].keys()):
            ret, frame_image = video_stream.read()
            frame_image = cv2.cvtColor(frame_image, cv2.COLOR_BGR2RGB)
            rows, cols = frame_image.shape[:2]
            bottom_left  = [int(cols * 0.05), int(rows * 0.8)]
            top_left     = [int(cols * 0.48), int(rows * 0.42)]
            bottom_right = [int(cols * 0.95), int(rows * 0.95)]
            top_right    = [int(cols * 0.55), int(rows * 0.4)]
    
            chips = []
            track_ids = []
            all_centroids[frame] = {}
            for i in range(len(annotations[video][frame]['challenge_object'])):
                if annotations[video][frame]['challenge_object'][i]['track_id'] not in challenge_object_centroids:
                    count_frame[annotations[video][frame]['challenge_object'][i]['track_id']] = 0
                    challenge_object_frames[annotations[video][frame]['challenge_object'][i]['track_id']] = []
                    challenge_object_centroids[annotations[video][frame]['challenge_object'][i]['track_id']] = []
                    challenge_object_minx[annotations[video][frame]['challenge_object'][i]['track_id']] = 999999999999
                    challenge_object_maxx[annotations[video][frame]['challenge_object'][i]['track_id']] = -1000000
                    
                x1, y1, x2, y2 = annotations[video][frame]['challenge_object'][i]['bbox']
                challenge_object_frames[annotations[video][frame]['challenge_object'][i]['track_id']].append(frame)
                challenge_object_centroids[annotations[video][frame]['challenge_object'][i]['track_id']].append([x1+(abs(x2-x1)/2),y1+(abs(y2-y1)/2)])
                challenge_object_minx[annotations[video][frame]['challenge_object'][i]['track_id']] = min(challenge_object_minx[annotations[video][frame]['challenge_object'][i]['track_id']], x1, x2)
                challenge_object_maxx[annotations[video][frame]['challenge_object'][i]['track_id']] = max(challenge_object_maxx[annotations[video][frame]['challenge_object'][i]['track_id']], x1, x2)
                chips.append( frame_image[max(0, int(1.2*y1-0.2*y2)):min(int(1.2*y2-0.2*y1), frame_image.shape[0]), max(0, int(1.2*x1-0.2*x2)):min(int(1.2*x2-0.2*x1), frame_image.shape[1])] )
                track_ids.append(annotations[video][frame]['challenge_object'][i]['track_id'])
                all_centroids[frame][annotations[video][frame]['challenge_object'][i]['track_id']] = [x1+(abs(x2-x1)/2),y1+(abs(y2-y1)/2)]
                count_frame[annotations[video][frame]['challenge_object'][i]['track_id']] += 1
                
                if annotations[video][frame]['challenge_object'][i]['track_id'] not in challenge_object_labels:
                    challenge_object_labels[annotations[video][frame]['challenge_object'][i]['track_id']] = {}
                    challenge_object_labels2[annotations[video][frame]['challenge_object'][i]['track_id']] = {}
    
            if len(chips) == 0:
                continue
            inputs = processor(images=chips, return_tensors="pt").to("cuda:1", dtype=torch.float16)
            predicted_class_idx = model(**inputs).logits.argmax(-1).detach().cpu().numpy().tolist()
            for track_id, label, image in zip(track_ids, predicted_class_idx, chips):
                label = imagenet_map(label)
                if label not in challenge_object_labels[track_id]:
                    challenge_object_labels[track_id][label] = 0.0
                challenge_object_labels[track_id][label] += image.shape[0]*image.shape[1]
    
            inputs = processor2(images=chips, return_tensors="pt").to("cuda:1", dtype=torch.float16)
            predicted_class_idx = model2(**inputs).logits.argmax(-1).detach().cpu().numpy().tolist()
            for track_id, label, image in zip(track_ids, predicted_class_idx, chips):
                label = imagenet_map(label)
                if label not in challenge_object_labels2[track_id]:
                    challenge_object_labels2[track_id][label] = 0.0
                challenge_object_labels2[track_id][label] += image.shape[0]*image.shape[1]
    
        hazards_prob = {k: [1.0, 0.0, 9999999, 0, 0, 1.0, 1.0, 1.0] for k in challenge_object_labels.keys()}
        # 1 car and center
        for k in challenge_object_labels.keys():
            if (max(challenge_object_labels[k].keys(), key=challenge_object_labels[k].get) == "car") and (challenge_object_labels[k]["car"]/sum(challenge_object_labels[k].values()) > 0.3):
                hazards_prob[k][0] = -10.0
            else:
                if "car" in challenge_object_labels[k]:
                    hazards_prob[k][0] = hazards_prob[k][0] - challenge_object_labels[k]["car"]/sum(challenge_object_labels[k].values())
    
        for frame in all_centroids.keys():
            track_ids = []
            centroids = []
            for k in all_centroids[frame].keys():
                if hazards_prob[k][0] > 0.2:
                    track_ids.append(k)
                    centroids.append(all_centroids[frame][k])
        
            if len(centroids) > 0:
                image_center = [frame_image.shape[1]/2, frame_image.shape[0]]
                potential_hazard_dists = np.linalg.norm(np.array(centroids)-image_center, axis=1)
                probable_hazard = np.argmin(potential_hazard_dists)
                hazards_prob[track_ids[probable_hazard]][2] = min(float(hazards_prob[track_ids[probable_hazard]][2]), potential_hazard_dists[probable_hazard])
    
        for k, v in hazards_prob.items():
            hazards_prob[k][2] = 1.0-float(hazards_prob[k][2]/max(x[2] for x in hazards_prob.values() if x != 9999999))
    
        # 2 left-right
        for k in challenge_object_frames:
            hazards_prob[k][1] = (challenge_object_maxx[k] - challenge_object_minx[k])/frame_image.shape[0] - 0.15*abs(challenge_object_centroids[k][0][0] - challenge_object_centroids[k][-1][0])/frame_image.shape[0]
        
        for k, v in hazards_prob.items():
            hazards_prob[k][3] = float(np.mean([is_point_in_quadrilateral(x, y, [bottom_left, top_left, bottom_right, top_right]) for x, y in challenge_object_centroids[k]])) -0.12*float(any([is_point_in_quadrilateral(x, y, [bottom_left, top_left, bottom_right, top_right]) for x, y in challenge_object_centroids[k]]))
    
        for k in challenge_object_frames:
            hazards_prob[k][4] = 1/len(challenge_object_frames.keys())
    
        for k in challenge_object_labels2.keys():
            if (max(challenge_object_labels2[k].keys(), key=challenge_object_labels2[k].get) == "car") and (challenge_object_labels2[k]["car"]/sum(challenge_object_labels2[k].values()) > 0.3):
                hazards_prob[k][5] = -10.0
            else:
                if "car" in challenge_object_labels2[k]:
                    hazards_prob[k][5] = hazards_prob[k][5] - challenge_object_labels2[k]["car"]/sum(challenge_object_labels2[k].values())
    
        for k in count_frame:
            hazards_prob[k][6] = 1- count_frame[k]/num_frames
        
        video_track_id[video] = []
        result = {}
        for k, v in hazards_prob.items():
            result[k] = sum( weight*np.random.normal(loc=1.0, scale=0.01, size=8)*np.array(v) ) # apply dp -> ensemble
    
        video_track_id[video] += [k for i, k in enumerate(sorted(result, key=result.get, reverse = True)) if (result[k] > 0  or i == 0)]
        Hazard_Track_video = [[-1]*num_frames for i in range(23)]
        for frame in range(num_frames):
            c = 0
            for track_id in video_track_id[video]:
                if track_id in [x['track_id'] for x in annotations[video][frame]['challenge_object']]:
                    Hazard_Track_video[c][frame] = track_id
                    if video not in video_first_hazard:
                        video_first_hazard[video] = frame
                    c += 1
                if c == 23:
                    print("out")
                    break
            
        for i in range(23):
            Hazard_Track_all[i] += Hazard_Track_video[i]
        print(f"'{video}': {video_track_id[video]}")
        ids += [f"{video}_{frame}" for frame in range(num_frames)]
        
    except:
        print(f'Error at {video}')
        continue

with open('video_track_id_tree.pkl', 'wb') as handle:
    pickle.dump(video_track_id, handle, protocol=pickle.HIGHEST_PROTOCOL)