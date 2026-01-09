import cv2
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# =========================
# CONFIG (GIỮ NGUYÊN)
# =========================
ANNOTATION_PATH = "/kaggle/input/coool-dataset/annotations_public.pkl"
VIDEO_PATH = "/kaggle/input/coool-dataset/COOOL-videos"
VIDEO_TRACK_1 = "video_track_id.pkl"
VIDEO_TRACK_2 = "video_track_id_tree.pkl"
OUTPUT_PKL = "hazard_name_by_id_blip2opt.pkl"

BATCH_SIZE = 128
DEVICE = "cuda"

# =========================
# MODEL
# =========================
processor = Blip2Processor.from_pretrained(
    "Salesforce/blip2-opt-6.7b",
    use_fast=False
)
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-6.7b",
    load_in_4bit=True
)
model.eval()

# =========================
# PROMPT & CLEANING
# =========================
PROMPT = "Describe the potential hazard in this driving scene using a short noun phrase."

REMOVE_WORDS = {
    "car","road","vehicle","street","driving","view","scene",
    "background","image","photo","front","camera"
}

def normalize_caption(text):
    text = text.lower().replace(",", "").replace(".", "")
    words = [w for w in text.split() if w not in REMOVE_WORDS]
    return " ".join(words[:6])

# =========================
# MAIN
# =========================
annotations = pickle.load(open(ANNOTATION_PATH, "rb"))
video_track_id = pickle.load(open(VIDEO_TRACK_1, "rb"))
video_track_id_tree = pickle.load(open(VIDEO_TRACK_2, "rb"))

hazard_name_by_id = {}

for video in tqdm(sorted(annotations.keys())):
    try:
        cap = cv2.VideoCapture(f"{VIDEO_PATH}/{video}.mp4")
        hazard_name_by_id.setdefault(video, {})

        batch_images, batch_ids = [], []

        for frame_idx in sorted(annotations[video].keys()):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            for obj in annotations[video][frame_idx]["challenge_object"]:
                tid = obj["track_id"]
                if tid not in video_track_id[video] + video_track_id_tree[video]:
                    continue

                x1,y1,x2,y2 = map(int, obj["bbox"])
                crop = frame[max(0,y1):min(y2,frame.shape[0]),
                             max(0,x1):min(x2,frame.shape[1])]
                if crop.size == 0:
                    continue

                batch_images.append(crop)
                batch_ids.append(tid)
                hazard_name_by_id[video].setdefault(tid, {})

            if len(batch_images) >= BATCH_SIZE:
                inputs = processor(
                    batch_images,
                    [PROMPT]*len(batch_images),
                    return_tensors="pt"
                ).to(DEVICE, torch.float16)

                outputs = model.generate(
                    **inputs,
                    max_length=20,
                    do_sample=False
                )

                captions = [
                    normalize_caption(
                        processor.decode(o, skip_special_tokens=True)
                    ) for o in outputs
                ]

                for tid, img, cap_text in zip(batch_ids, batch_images, captions):
                    if cap_text == "":
                        continue
                    hazard_name_by_id[video][tid][cap_text] = \
                        hazard_name_by_id[video][tid].get(cap_text, 0.0) + img.shape[0]*img.shape[1]

                batch_images, batch_ids = [], []

        cap.release()

    except Exception as e:
        print(f"Error at {video}: {e}")
        continue

with open(OUTPUT_PKL, "wb") as f:
    pickle.dump(hazard_name_by_id, f, protocol=pickle.HIGHEST_PROTOCOL)
