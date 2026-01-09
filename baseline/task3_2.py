import cv2
import torch
import pickle
import numpy as np
import pandas as pd

from tqdm import *
from transformers import Blip2Processor, Blip2ForConditionalGeneration

batchsize = 256
device = "cuda"

df_final = pd.DataFrame()
annotations = pickle.load(open("/kaggle/input/coool-dataset/annotations_public.pkl", 'rb'))
video_track_id = pickle.load(open("video_track_id.pkl", 'rb'))
video_track_id_tree = pickle.load(open("video_track_id_tree.pkl", 'rb'))

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-6.7b", use_fast=False)
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-6.7b", load_in_4bit=True)
model.eval()

hazard_name_by_id = {}
def clean_text(text):
    text = text.replace("car view of ", "").replace(",", "").split()
    i = 1
    while i !=  len(text):
        if (text[i] == text[i-1]) or (text[i] == ""):
            text.pop(i)
        else:
            i += 1
    text[0] = text[0][0].upper() + text[0][1:]
    return " ".join(text)

for video in tqdm(sorted(annotations.keys())):
    try:
        video_stream = cv2.VideoCapture(f"/kaggle/input/coool-dataset/COOOL-videos/{video}.mp4")
        num_frames = len(annotations[video].keys())
        batch_image = []
        if video not in hazard_name_by_id:
            hazard_name_by_id[video] = {}
        for frame in sorted(annotations[video].keys()):
            ret, frame_image = video_stream.read()
            frame_image = cv2.cvtColor(frame_image, cv2.COLOR_BGR2RGB)
    
            for i in range(len(annotations[video][frame]['challenge_object'])):
                if annotations[video][frame]['challenge_object'][i]['track_id'] in video_track_id[video] + video_track_id_tree[video] :
                    batch_image.append( frame_image )
                    break
    
            if ((len(batch_image) >= batchsize) or (frame == num_frames-1)) and len(batch_image) > 0:
                inputs = processor(batch_image, ["car view of"]*len(batch_image), return_tensors="pt").to(device, torch.float16)
                output = model.generate(**inputs, max_length=64)
                output_text = [clean_text(processor.decode(output[i], skip_special_tokens=True)) for i in range(len(batch_image))]
    
                for image, text in zip(batch_image, output_text):
                    if text not in hazard_name_by_id[video]:
                        hazard_name_by_id[video][text] = 0.0
                    hazard_name_by_id[video][text] += image.shape[0]*image.shape[1]
                
                batch_image = []
    except Exception as e:
        print(f"Error at {video}: {e}")
        continue

with open("hazard_name_by_frame_blip2opt.pkl", "wb") as handle:
    pickle.dump(hazard_name_by_id, handle, protocol=pickle.HIGHEST_PROTOCOL)