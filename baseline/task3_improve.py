import cv2
import torch
import pickle
import numpy as np

from tqdm import tqdm
from PIL import Image
from collections import defaultdict

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
import clip

# ======================================================
# CONFIG
# ======================================================
ANNOT_PATH = "/kaggle/input/coool-dataset/annotations_public.pkl"
VIDEO_DIR  = "/kaggle/input/coool-dataset/COOOL-videos"

MODEL_NAME = "OpenGVLab/InternVL3_5-8B"
DEVICE = "cuda"

FRAME_STRIDE = 5        # caption mỗi 5 frame / track
TOPK_SYNONYM = 8

# ======================================================
# LOAD DATA
# ======================================================
annotations = pickle.load(open(ANNOT_PATH, "rb"))
video_track_id = pickle.load(open("video_track_id.pkl", "rb"))
video_track_id_tree = pickle.load(open("video_track_id_tree.pkl", "rb"))

# ======================================================
# LOAD INTERNVL
# ======================================================
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
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

# ======================================================
# LOAD CLIP
# ======================================================
clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
clip_model.eval()

# ======================================================
# PROMPT
# ======================================================
PROMPT = (
    "<image>\n"
    "Describe the dangerous object in front of the car using short phrases."
)

GEN_CFG = dict(
    max_new_tokens=32,
    do_sample=False,
    temperature=0.0,
    eos_token_id=151645,
    pad_token_id=151645
)

# ======================================================
# IMAGE PREPROCESS
# ======================================================
transform = T.Compose([
    T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
])

def preprocess_image(np_img):
    return transform(Image.fromarray(np_img))\
        .unsqueeze(0)\
        .to(torch.float16)\
        .cuda(non_blocking=True)

# ======================================================
# UTILS
# ======================================================
def clean_text(text):
    text = text.replace(",", "").strip()
    return text[:1].upper() + text[1:] if text else text

def motion_weight(area, prev_area, frame_gap, tau=6.0, gamma=12.0):
    if prev_area is None:
        return 1.0
    growth = max((area - prev_area) / (prev_area + 1e-6), 0)
    ttc = 1.0 / (growth + 1e-3)
    return float(np.exp(-ttc / tau) * np.exp(-frame_gap / gamma))

@torch.no_grad()
def merge_synonyms(score_dict, topk=TOPK_SYNONYM, thresh=0.85):
    if len(score_dict) <= 1:
        return score_dict

    items = sorted(score_dict.items(), key=lambda x: -x[1])[:topk]
    keys = [k for k,_ in items]
    scores = np.array([v for _,v in items])

    tokens = clip.tokenize(keys).to(DEVICE)
    emb = clip_model.encode_text(tokens)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    emb = emb.cpu().numpy()

    merged = {}
    used = set()
    for i,k in enumerate(keys):
        if i in used:
            continue
        total = scores[i]
        used.add(i)
        for j in range(i+1,len(keys)):
            if j not in used and emb[i] @ emb[j] > thresh:
                total += scores[j]
                used.add(j)
        merged[k] = float(total)
    return merged

# ======================================================
# OUTPUT
# ======================================================
hazard_name_by_id = {}
hazard_name_by_frame = {}

# ======================================================
# MAIN LOOP
# ======================================================
for video in tqdm(sorted(annotations.keys())[:2]):
    try:
        cap = cv2.VideoCapture(f"{VIDEO_DIR}/{video}.mp4")
        hazard_name_by_id.setdefault(video, {})
        hazard_name_by_frame.setdefault(video, {})

        prev_area = {}
        prev_frame = {}
        caption_cache = {}   # (tid or frame_idx) -> text

        for frame_idx in sorted(annotations[video].keys()):
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            has_hazard = False

            # ---------- OBJECT LEVEL ----------
            for obj in annotations[video][frame_idx]["challenge_object"]:
                tid = obj["track_id"]
                if tid not in (video_track_id[video] + video_track_id_tree[video]):
                    continue

                if tid in prev_frame and frame_idx - prev_frame[tid] < FRAME_STRIDE:
                    continue

                x1,y1,x2,y2 = obj["bbox"]
                crop = frame[
                    max(0,int(1.1*y1-0.1*y2)):min(int(1.1*y2-0.1*y1),frame.shape[0]),
                    max(0,int(1.1*x1-0.1*x2)):min(int(1.1*x2-0.1*x1),frame.shape[1])
                ]
                if crop.size == 0:
                    continue

                has_hazard = True
                hazard_name_by_id[video].setdefault(tid, {})

                if tid not in caption_cache:
                    pv = preprocess_image(crop)
                    txt = model.chat(tokenizer, pv, PROMPT, GEN_CFG)
                    caption_cache[tid] = clean_text(txt)

                txt = caption_cache[tid]
                area = crop.shape[0] * crop.shape[1]
                w = motion_weight(area, prev_area.get(tid), frame_idx - prev_frame.get(tid, frame_idx))

                hazard_name_by_id[video][tid][txt] = \
                    hazard_name_by_id[video][tid].get(txt, 0.0) + area * w

                prev_area[tid] = area
                prev_frame[tid] = frame_idx

            # ---------- FRAME LEVEL ----------
            if has_hazard and frame_idx % FRAME_STRIDE == 0:
                if frame_idx not in caption_cache:
                    pv = preprocess_image(frame)
                    txt = model.chat(tokenizer, pv, PROMPT, GEN_CFG)
                    caption_cache[frame_idx] = clean_text(txt)

                txt = caption_cache[frame_idx]
                hazard_name_by_frame[video][txt] = \
                    hazard_name_by_frame[video].get(txt, 0.0) + frame.shape[0]*frame.shape[1]

        # ---------- MERGE SYNONYM ----------
        for tid in hazard_name_by_id[video]:
            hazard_name_by_id[video][tid] = merge_synonyms(
                hazard_name_by_id[video][tid]
            )

    except Exception as e:
        print(f"Error at {video}: {e}")
        continue

# ======================================================
# SAVE
# ======================================================
with open("hazard_name_by_id.pkl", "wb") as f:
    pickle.dump(hazard_name_by_id, f)

with open("hazard_name_by_frame.pkl", "wb") as f:
    pickle.dump(hazard_name_by_frame, f)

print("✅ TASK 3 FIXED VERSION FINISHED")