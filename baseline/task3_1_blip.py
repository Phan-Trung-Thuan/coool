import cv2
import torch
import pickle
import numpy as np
import pandas as pd

from tqdm import *
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BlipProcessor, BlipForConditionalGeneration, VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from clip_interrogator import Config, Interrogator

batchsize = 512
df_final = pd.DataFrame()
annotations = pickle.load(open("/kaggle/input/coool-dataset/annotations_public.pkl", 'rb'))
video_track_id = pickle.load(open("video_track_id.pkl", 'rb'))
video_track_id_tree = pickle.load(open("video_track_id_tree.pkl", 'rb'))

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large", use_fast=False)
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", load_in_4bit=True)

hazard_name_by_id = {}
# hazard_name_by_id = pickle.load(open("hazard_name_by_id_final.pkl", "rb"))
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

for video in tqdm(sorted(annotations.keys())[120:]):
    try:
        video_stream = cv2.VideoCapture(f"/kaggle/input/coool-dataset/COOOL-videos/{video}.mp4")
        
        num_frames = len(annotations[video].keys())
        batch_id = []
        batch_image = []
        if video not in hazard_name_by_id:
            hazard_name_by_id[video] = {}
        for frame in sorted(annotations[video].keys()):
            ret, frame_image = video_stream.read()
            frame_image = cv2.cvtColor(frame_image, cv2.COLOR_BGR2RGB)
            
            for i in range(len(annotations[video][frame]['challenge_object'])):
                if annotations[video][frame]['challenge_object'][i]['track_id'] in video_track_id[video] + video_track_id_tree[video] :
                    x1, y1, x2, y2 = annotations[video][frame]['challenge_object'][i]['bbox']
                    batch_id.append(annotations[video][frame]['challenge_object'][i]['track_id'])
                    batch_image.append( frame_image[max(0, int(1.1*y1-0.1*y2)):min(int(1.1*y2-0.1*y1), frame_image.shape[0]), max(0, int(1.1*x1-0.1*x2)):min(int(1.1*x2-0.1*x1), frame_image.shape[1])] )
                    if annotations[video][frame]['challenge_object'][i]['track_id'] not in hazard_name_by_id[video]:
                        hazard_name_by_id[video][annotations[video][frame]['challenge_object'][i]['track_id']] = {}
                    
            if ((len(batch_id) >= batchsize) or (frame == num_frames-1)) and len(batch_id) > 0:
                inputs = processor(batch_image, ["car view of"]*len(batch_image), return_tensors="pt").to("cuda", torch.float16)
                output = model.generate(**inputs, max_length=64)
                output_text = [clean_text(processor.decode(output[i], skip_special_tokens=True)) for i in range(len(batch_id))]
    
                for track_id, image, text in zip(batch_id, batch_image, output_text):
                    if text not in hazard_name_by_id[video][track_id]:
                        hazard_name_by_id[video][track_id][text] = 0.0
                    hazard_name_by_id[video][track_id][text] += image.shape[0]*image.shape[1]
                
                batch_id = []
                batch_image = []
        
    except Exception as e:
        print(f'Error at {video}')
        print(e)
        continue

with open('hazard_name_by_id_blip.pkl', 'wb') as handle:
    pickle.dump(hazard_name_by_id, handle, protocol=pickle.HIGHEST_PROTOCOL)