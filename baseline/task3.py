import cv2
import torch
import pickle
import numpy as np
import pandas as pd

from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# =========================
# CONFIG
# =========================
BATCHSIZE = 128        # task 3.1
DEVICE = "cuda"

# =========================
# LOAD DATA
# =========================
annotations = pickle.load(open("/kaggle/input/coool-dataset/annotations_public.pkl", "rb"))
video_track_id = pickle.load(open("video_track_id.pkl", "rb"))
video_track_id_tree = pickle.load(open("video_track_id_tree.pkl", "rb"))

# =========================
# LOAD MODEL
# =========================
processor = Blip2Processor.from_pretrained(
    "Salesforce/blip2-opt-6.7b", use_fast=False
)
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-6.7b", load_in_4bit=True
)
model.eval()

# =========================
# OUTPUT CONTAINERS
# =========================
hazard_name_by_id = {}       # task 3.1
hazard_name_by_frame = {}    # task 3.2

# =========================
# UTILS
# =========================
def clean_text(text):
    text = text.replace("car view of ", "").replace(",", "").split()
    i = 1
    while i < len(text):
        if text[i] == text[i - 1] or text[i] == "":
            text.pop(i)
        else:
            i += 1
    text[0] = text[0][0].upper() + text[0][1:]
    return " ".join(text)

# =========================
# MAIN LOOP
# =========================
for video in tqdm(sorted(annotations.keys())):
    try:
        video_stream = cv2.VideoCapture(
            f"/kaggle/input/coool-dataset/COOOL-videos/{video}.mp4"
        )
        num_frames = len(annotations[video])

        # init outputs
        hazard_name_by_id.setdefault(video, {})
        hazard_name_by_frame.setdefault(video, {})

        # buffers
        batch_id, batch_img_id = [], []
        batch_img_frame = []

        for frame_idx in sorted(annotations[video].keys()):
            ret, frame_image = video_stream.read()
            if not ret:
                continue

            frame_image = cv2.cvtColor(frame_image, cv2.COLOR_BGR2RGB)

            # =========================
            # SCAN OBJECTS
            # =========================
            for obj in annotations[video][frame_idx]["challenge_object"]:
                track_id = obj["track_id"]

                if track_id not in (video_track_id[video] + video_track_id_tree[video]):
                    continue

                x1, y1, x2, y2 = obj["bbox"]

                # -------- task 3.1 (crop by track_id) --------
                crop = frame_image[
                    max(0, int(1.1 * y1 - 0.1 * y2)) : min(int(1.1 * y2 - 0.1 * y1), frame_image.shape[0]),
                    max(0, int(1.1 * x1 - 0.1 * x2)) : min(int(1.1 * x2 - 0.1 * x1), frame_image.shape[1]),
                ]

                batch_id.append(track_id)
                batch_img_id.append(crop)

                hazard_name_by_id[video].setdefault(track_id, {})

                # -------- task 3.2 (full frame) --------
                batch_img_frame.append(frame_image)
                break  # chỉ cần 1 hazard per frame cho task 3.2

            # =========================
            # RUN TASK 3.1
            # =========================
            if (len(batch_id) >= BATCHSIZE) or (frame_idx == num_frames - 1):
                if len(batch_id) > 0:
                    inputs = processor(
                        batch_img_id,
                        ["car view of"] * len(batch_img_id),
                        return_tensors="pt",
                    ).to(DEVICE, torch.float16)

                    outputs = model.generate(**inputs, max_length=64)
                    texts = [
                        clean_text(processor.decode(o, skip_special_tokens=True))
                        for o in outputs
                    ]

                    for tid, img, txt in zip(batch_id, batch_img_id, texts):
                        hazard_name_by_id[video][tid].setdefault(txt, 0.0)
                        hazard_name_by_id[video][tid][txt] += img.shape[0] * img.shape[1]

                batch_id, batch_img_id = [], []

            # =========================
            # RUN TASK 3.2
            # =========================
            if (len(batch_img_frame) >= BATCHSIZE) or (frame_idx == num_frames - 1):
                if len(batch_img_frame) > 0:
                    inputs = processor(
                        batch_img_frame,
                        ["car view of"] * len(batch_img_frame),
                        return_tensors="pt",
                    ).to(DEVICE, torch.float16)

                    outputs = model.generate(**inputs, max_length=64)
                    texts = [
                        clean_text(processor.decode(o, skip_special_tokens=True))
                        for o in outputs
                    ]

                    for img, txt in zip(batch_img_frame, texts):
                        hazard_name_by_frame[video].setdefault(txt, 0.0)
                        hazard_name_by_frame[video][txt] += img.shape[0] * img.shape[1]

                batch_img_frame = []

    except Exception as e:
        print(f"Error at {video}: {e}")
        continue

# =========================
# SAVE OUTPUTS (UNCHANGED)
# =========================
with open("hazard_name_by_id_blip2opt.pkl", "wb") as f:
    pickle.dump(hazard_name_by_id, f, protocol=pickle.HIGHEST_PROTOCOL)

with open("hazard_name_by_frame_blip2opt.pkl", "wb") as f:
    pickle.dump(hazard_name_by_frame, f, protocol=pickle.HIGHEST_PROTOCOL)
