import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

import heapq
import string
import pickle
import numpy as np
import pandas as pd

from tqdm.notebook import *
from collections import Counter

df_final = pd.DataFrame()
annotations = pickle.load(open("/kaggle/input/coool-dataset/annotations_public.pkl", 'rb'))
task1 = pickle.load(open("video_Driver_State_Changed.pkl", "rb"))
task2_1 = pickle.load(open("video_track_id.pkl", "rb"))
task2_2 = pickle.load(open("video_track_id_tree.pkl", "rb"))
task3s = [
    # pickle.load(open("hazard_name_by_id_blip2opt.pkl", "rb")),
    # pickle.load(open("hazard_name_by_id_blip2flan.pkl", "rb")),
    # pickle.load(open("hazard_name_by_id_blip.pkl", "rb")),
    # pickle.load(open("hazard_name_by_id_vit.pkl", "rb")),
    pickle.load(open("../v1/hazard_name_by_id.pkl", "rb")),
         ]
# task3_frame = pickle.load(open("hazard_name_by_frame_blip2opt.pkl", "rb"))
task3_frame = pickle.load(open("../v1/hazard_name_by_frame.pkl", "rb"))
task3 = {}
# w = np.array([1.26830, 0.03213, 0.01384, 0.00016])
w = np.array([1.26830])

def mix(d, frame):
    new_d = {}
    for track_id, text_score in d.items():
        new_d[track_id] = d[track_id].copy()
        for other_track_id, text_score in d.items():
            for text, score in text_score.items():
                if text not in new_d[track_id]:
                    new_d[track_id][text] = 0.0
                new_d[track_id][text] += score/32.1098
        for text, score in frame.items():
            if text not in new_d[track_id]:
                new_d[track_id][text] = 0.0
            new_d[track_id][text] += score/(10*1920*1080)
    return new_d

for i in range(len(task3s)):
    for video in sorted(list(annotations.keys())):
        if video not in task3:
            task3[video] = {}
        try:
            for track_id, name_score in task3s[i][video].items():
                if track_id not in task3[video]:
                    task3[video][track_id] = {}
                for name, score in name_score.items():
                    if name not in task3[video][track_id]:
                        task3[video][track_id][name] = 0.0
                    task3[video][track_id][name] += w[i]*score
        except:
            print(i, video)

for video in tqdm(sorted(list(annotations.keys()))):
    task3[video] = mix(task3[video], task3_frame[video])

remove = ["a", "the", "street", "walking", "on", "and", "with", "in", "of", "blurry", "road", "crossing", "background", "sitting", "foreground", "photo", "image", "running", "line", "down", "highway", "up", "front", "rain", "across", "driving", "at", "daytime", "night", "standing", "air", "through", "pickup", "day", "has", "roof", "driveway", "ford", "explorer", "her", "covered", "snow", "snowy", "water", "small", "sky", "over", "flying", "ha", "posing", "poses", 
          "cross", "is", "ground", "parking", "parked", "s", "out", "from", "by", "it", "other", "riding", "laptop", "computer", "keyboard", "television", "window", "lamp", "its", "his", "new", "picture", "city", "dmax", "bathroom", "king", "moon", "ufo", "suspect", "shirt", "object", "st", "johns", "logo", "thomas", "edward", "hitting", "mirror", "doing", "hazard", "dashcam", "shows", "this", "that", "middle", "presence", "which", "no", "haz", "there", "lot", "large", 
          "car", "drivers", "toyota", "yaris", "sidewalk", "mazda", 
         ]

reduce = ["white", "black", "yellow", "dark", "gray", "brown", "green", "accident", "two"] # new

def clean_text(s):
    s = s.lower()
    s = "".join([c for c in s if c in string.ascii_lowercase+" "])
    return s

def is_noun(word):
    word_tokenized = nltk.word_tokenize(word)
    pos = nltk.pos_tag(word_tokenized)[0][1]
    return pos in ['NN', 'NNS', 'NNP', 'NNPS']

def is_animal(word):
    synsets = nltk.corpus.wordnet.synsets(word)
    for synset in synsets:
        if ('animal' in synset.lexname()) or ('mammal' in synset.lexname()) or ('person' in synset.lexname()) or ('person' in word):
            return True
    return False

def clean_all_words(d):
    for track_id, text_score in d.items():
        clean_text_score = {}
        for text, score in text_score.items():
            for w in clean_text(text).split():
                if w not in clean_text_score:
                    clean_text_score[w] = 0.0
                clean_text_score[w] += score
        for w in remove:
            if w in clean_text_score:
                clean_text_score.pop(w)

        for w in reduce:
            if w in clean_text_score:
                clean_text_score[w] = clean_text_score[w]/3
                
        for w in clean_text_score:
            if is_noun(w):
                clean_text_score[w] = 2*clean_text_score[w]
            if is_animal(w):
                clean_text_score[w] = 1.5*clean_text_score[w]
        d[track_id] = {k: v for k, v in sorted(clean_text_score.items(), key=lambda item: item[1], reverse = True)}
    return d

def clean35(s, max_word = 20):
    if len(s) == 0:
        return -1
    c = 0
    i = 0
    r = []
    while (c <= 100) and (i<min(len(s), max_word+1)):
        r.append(s[i])
        c += len(s[i]) + 1
        i += 1
    return " ".join(r[:-1])

ids = []
Driver_State_Changed = []
Hazard_Track_all = [[] for i in range(23)]
Hazard_Track_name_all = [[] for i in range(23)]

for video in tqdm(sorted(list(annotations.keys()))):
    try:
        # Checking
        annotations[video]
        task1[video]
        task2_1[video] + task2_2[video]
        task3[video]
        
        num_frames = len(annotations[video])
        Hazard_Track_video = [[-1]*num_frames for i in range(23)]
        Hazard_Track_name_video = [[-1]*num_frames for i in range(23)]
        ids += [f"{video}_{frame}" for frame in range(num_frames)]
        hazard_appear = False
        task3[video] = clean_all_words(task3[video])
        for frame in sorted(annotations[video].keys()):
            c = 0
            for track_id in Counter(task2_1[video] + task2_2[video]).keys():
                if track_id in [x['track_id'] for x in annotations[video][frame]['challenge_object']]:
                    hazard_appear = True
                    Hazard_Track_video[c][frame] = track_id
                    if track_id in task3[video]:
                        Hazard_Track_name_video[c][frame] = clean35(sorted(task3[video][track_id], key=task3[video][track_id].get, reverse=True)[:36])
                    else:
                        Hazard_Track_name_video[c][frame] = " "
                    c += 1
            if (frame >= task1[video][1]):
                if not hazard_appear and task1[video][0] == 'random':
                    Driver_State_Changed.append(False)
                else:
                    Driver_State_Changed.append(True)
            else:
                Driver_State_Changed.append(False)
                
        for i in range(23):
            Hazard_Track_all[i] += Hazard_Track_video[i]
            Hazard_Track_name_all[i] += Hazard_Track_name_video[i]
    except:
        print(f'Skipp video {video}')
        continue

df_final["ID"] = ids
df_final["Driver_State_Changed"] = Driver_State_Changed

for i in range(23):
    df_final[f"Hazard_Track_{i}"] = -1
    df_final[f"Hazard_Name_{i}"] = ' '
    
for i in range(23):
    df_final[f"Hazard_Track_{i}"] = Hazard_Track_all[i]
    df_final[f"Hazard_Name_{i}"] = Hazard_Track_name_all[i]

df_final.to_csv("final.csv", index=False)