import cv2
import torch
import pickle
import numpy as np
import pandas as pd

from tqdm import tqdm
from collections import defaultdict
from PIL import Image

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
import clip

annotations = pickle.load(open("/kaggle/input/coool-dataset/annotations_public.pkl", "rb"))
video_track_id = pickle.load(open("video_track_id.pkl", "rb"))
video_track_id_tree = pickle.load(open("video_track_id_tree.pkl", "rb"))

MODEL_NAME = "OpenGVLab/InternVL3_5-8B"
BATCHSIZE = 128

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    use_fast=False
)

model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    device_map="auto",
    quantization_config=bnb_cfg,
    low_cpu_mem_usage=True
).eval()

clip_model, _ = clip.load("ViT-B/32", device="cuda")
clip_model.eval()

PROMPT = (
    "<image>\n"
    "Describe the main object that may cause danger to a car. "
    "Use a short phrase like 'A cat crossing the road', "
    "'Moose running across the street', 'Dog on road'."
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

transform = T.Compose([
    T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def preprocess_image(np_img):
    img = Image.fromarray(np_img)
    return transform(img).unsqueeze(0).to(torch.float16).cuda()

def clean_text(text):
    text = text.replace(",", "").strip()
    text = text[0].upper() + text[1:] if len(text) > 0 else text
    return text

def motion_weight(area, prev_area, frame_gap, tau=6.0, gamma=12.0):
    if prev_area is None:
        return 1.0
    growth = max((area - prev_area) / (prev_area + 1e-6), 0)
    ttc = 1.0 / (growth + 1e-3)
    return np.exp(-ttc / tau) * np.exp(-frame_gap / gamma)

@torch.no_grad()
def merge_synonyms(score_dict, thresh=0.85):
    if len(score_dict) <= 1:
        return score_dict

    keys = list(score_dict.keys())
    scores = np.array([score_dict[k] for k in keys])

    tokens = clip.tokenize(keys).cuda()
    emb = clip_model.encode_text(tokens)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    emb = emb.cpu().numpy()

    used = set()
    merged = {}

    for i, k in enumerate(keys):
        if i in used:
            continue
        total = scores[i]
        used.add(i)
        for j in range(i + 1, len(keys)):
            if j not in used and emb[i] @ emb[j] > thresh:
                total += scores[j]
                used.add(j)
        merged[k] = float(total)

    return merged

hazard_name_by_id = {}
hazard_name_by_frame = {}

for video in tqdm(sorted(annotations.keys())[100:]):
    try:
        video_stream = cv2.VideoCapture(
            f"/kaggle/input/coool-dataset/COOOL-videos/{video}.mp4"
        )

        num_frames = len(annotations[video])

        hazard_name_by_id.setdefault(video, {})
        hazard_name_by_frame.setdefault(video, {})

        batch_id, batch_img_id = [], []
        batch_img_frame = []

        prev_area = {}
        prev_frame = {}

        for frame_idx in sorted(annotations[video].keys()):
            ret, frame_image = video_stream.read()
            if not ret:
                continue

            frame_image = cv2.cvtColor(frame_image, cv2.COLOR_BGR2RGB)
            has_hazard_in_frame = False

            for obj in annotations[video][frame_idx]["challenge_object"]:
                track_id = obj["track_id"]
                if track_id not in (video_track_id[video] + video_track_id_tree[video]):
                    continue

                x1, y1, x2, y2 = obj["bbox"]
                crop = frame_image[
                    max(0, int(1.1*y1 - 0.1*y2)):min(int(1.1*y2 - 0.1*y1), frame_image.shape[0]),
                    max(0, int(1.1*x1 - 0.1*x2)):min(int(1.1*x2 - 0.1*x1), frame_image.shape[1])
                ]
                if crop.size == 0:
                    continue

                batch_id.append(track_id)
                batch_img_id.append(crop)
                hazard_name_by_id[video].setdefault(track_id, {})
                has_hazard_in_frame = True

            if has_hazard_in_frame:
                batch_img_frame.append(frame_image)

            # -------- TASK 3.1 --------
            if ((len(batch_id) >= BATCHSIZE) or (frame_idx == num_frames - 1)) and batch_id:
                for tid, img in zip(batch_id, batch_img_id):
                    pv = preprocess_image(img)
                    generation_config = dict(max_new_tokens=64, do_sample=True, eos_token_id=151645,pad_token_id=151645)
                    txt = model.chat(tokenizer, pv, PROMPT, generation_config)
                    txt = clean_text(txt)

                    area = img.shape[0] * img.shape[1]
                    w = motion_weight(
                        area,
                        prev_area.get(tid),
                        frame_idx - prev_frame.get(tid, frame_idx)
                    )

                    hazard_name_by_id[video][tid][txt] = (
                        hazard_name_by_id[video][tid].get(txt, 0.0)
                        + area * w
                    )

                    prev_area[tid] = area
                    prev_frame[tid] = frame_idx

                batch_id, batch_img_id = [], []

            # -------- TASK 3.2 --------
            if ((len(batch_img_frame) >= BATCHSIZE) or (frame_idx == num_frames - 1)) and batch_img_frame:
                for img in batch_img_frame:
                    pv = preprocess_image(img)
                    generation_config = dict(max_new_tokens=64, do_sample=True, eos_token_id=151645,pad_token_id=151645)
                    txt = model.chat(tokenizer, pv, PROMPT, generation_config)
                    txt = clean_text(txt)

                    hazard_name_by_frame[video][txt] = (
                        hazard_name_by_frame[video].get(txt, 0.0)
                        + img.shape[0] * img.shape[1]
                    )
                batch_img_frame = []

        # ---- MERGE SYNONYM SAU VIDEO ----
        for tid in hazard_name_by_id[video]:
            hazard_name_by_id[video][tid] = merge_synonyms(hazard_name_by_id[video][tid])

    except Exception as e:
        print(f"Error at {video}: {e}")
        continue

with open("hazard_name_by_id.pkl", "wb") as f:
    pickle.dump(hazard_name_by_id, f, protocol=pickle.HIGHEST_PROTOCOL)

with open("hazard_name_by_frame.pkl", "wb") as f:
    pickle.dump(hazard_name_by_frame, f, protocol=pickle.HIGHEST_PROTOCOL)
