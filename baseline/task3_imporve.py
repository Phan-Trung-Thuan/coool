# =========================
# task3_internvl3.py
# =========================

import os
import cv2
import torch
import pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

# =========================
# CONFIG
# =========================
VIDEO_DIR = "/kaggle/input/coool-dataset/COOOL-videos"
ANNOT_PATH = "/kaggle/input/coool-dataset/annotations_public.pkl"

OUT_ID = "hazard_name_by_id.pkl"
OUT_FRAME = "hazard_name_by_frame.pkl"

DEVICE = "cuda"

# =========================
# LOAD INTERNVL3 14B 4BIT
# =========================
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

MODEL_NAME = "OpenGVLab/InternVL3-14B"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    device_map="auto",
    quantization_config=bnb_cfg
)
model.eval()

# =========================
# PROMPT
# =========================
SYSTEM_PROMPT = (
    "You are an autonomous driving safety system.\n"
    "Given an image, output ONLY the main road hazard.\n"
    "Rules:\n"
    "- Use at most 3 words\n"
    "- Use a noun phrase\n"
    "- No adjectives\n"
    "- No verbs\n"
    "- No punctuation\n"
    "- If nothing dangerous: output 'none'\n"
)

def infer_hazard(image):
    prompt = "<image>\n" + SYSTEM_PROMPT

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text.strip().lower()

# =========================
# MAIN
# =========================
annotations = pickle.load(open(ANNOT_PATH, "rb"))

hazard_name_by_id = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
hazard_name_by_frame = defaultdict(lambda: defaultdict(float))

for video in tqdm(sorted(annotations.keys())):
    cap = cv2.VideoCapture(f"{VIDEO_DIR}/{video}.mp4")
    num_frames = len(annotations[video])

    track_hazards = defaultdict(list)
    frame_hazards = defaultdict(list)

    for frame_id in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        for obj in annotations[video][frame_id]["challenge_object"]:
            tid = obj["track_id"]
            x1, y1, x2, y2 = obj["bbox"]

            crop = frame[
                max(0, int(y1)):min(frame.shape[0], int(y2)),
                max(0, int(x1)):min(frame.shape[1], int(x2))
            ]

            if crop.size == 0:
                continue

            hazard = infer_hazard(crop)
            if hazard == "none":
                continue

            track_hazards[tid].append(hazard)
            frame_hazards[frame_id].append(hazard)

    # =========================
    # AGGREGATION (TASK 3-1)
    # =========================
    for tid, hz_list in track_hazards.items():
        counter = Counter(hz_list)
        total = sum(counter.values())
        for h, c in counter.items():
            hazard_name_by_id[video][tid][h] = c / total

    # =========================
    # AGGREGATION (TASK 3-2)
    # =========================
    for fid, hz_list in frame_hazards.items():
        counter = Counter(hz_list)
        total = sum(counter.values())
        for h, c in counter.items():
            hazard_name_by_frame[f"{video}_{fid}"][h] = c / total

# =========================
# SAVE (UNCHANGED FORMAT)
# =========================
with open(OUT_ID, "wb") as f:
    pickle.dump(hazard_name_by_id, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(OUT_FRAME, "wb") as f:
    pickle.dump(hazard_name_by_frame, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Task 3 completed with InternVL3-14B (4bit)")
