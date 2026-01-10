import cv2
import torch
import pickle
import numpy as np

from tqdm import tqdm
from collections import defaultdict
from PIL import Image

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
import clip

# =========================
# CONFIG
# =========================
ANNOT_PATH = "/kaggle/input/coool-dataset/annotations_public.pkl"
VIDEO_DIR = "/kaggle/input/coool-dataset/COOOL-videos"

MODEL_NAME = "OpenGVLab/InternVL3_5-8B"
DEVICE = "cuda"

BATCHSIZE = 64
FRAME_STRIDE = 10        # caption mỗi 10 frame / track
TOPK_SYNONYM = 8

# =========================
# LOAD DATA
# =========================
annotations = pickle.load(open(ANNOT_PATH, "rb"))
video_track_id = pickle.load(open("video_track_id.pkl", "rb"))
video_track_id_tree = pickle.load(open("video_track_id_tree.pkl", "rb"))

# =========================
# LOAD INTERNVL (FAST CONFIG)
# =========================
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
    low_cpu_mem_usage=True,
    use_flash_attn=False
).eval()

# =========================
# LOAD CLIP (FOR SYNONYM MERGE)
# =========================
clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
clip_model.eval()

# =========================
# PROMPT (MATCH LB DISTRIBUTION)
# =========================
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

# =========================
# IMAGE PREPROCESS (FAST)
# =========================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

transform = T.Compose([
    T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def preprocess_image(np_img):
    img = Image.fromarray(np_img)
    return transform(img).unsqueeze(0).to(torch.float16).cuda(non_blocking=True)

# =========================
# UTILS
# =========================
def clean_text(text):
    text = text.replace(",", "").strip()
    if len(text) > 0:
        text = text[0].upper() + text[1:]
    return text

def motion_weight(area, prev_area, frame_gap, tau=6.0, gamma=12.0):
    if prev_area is None:
        return 1.0
    growth = max((area - prev_area) / (prev_area + 1e-6), 0)
    ttc = 1.0 / (growth + 1e-3)
    w = np.exp(-ttc / tau) * np.exp(-frame_gap / gamma)
    return float(np.clip(w, 0.2, 3.0))

@torch.no_grad()
def merge_synonyms_fast(score_dict, topk=TOPK_SYNONYM, thresh=0.85):
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

# =========================
# OUTPUT
# =========================
hazard_name_by_id = {}
hazard_name_by_frame = {}

# =========================
# MAIN LOOP (FAST)
# =========================
for video in tqdm(sorted(annotations.keys())):
    try:
        cap = cv2.VideoCapture(f"{VIDEO_DIR}/{video}.mp4")
        num_frames = len(annotations[video])

        hazard_name_by_id.setdefault(video, {})
        hazard_name_by_frame.setdefault(video, {})

        prev_area = {}
        prev_frame = {}
        caption_cache = {}

        for frame_idx in sorted(annotations[video].keys()):
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            has_hazard = False

            for obj in annotations[video][frame_idx]["challenge_object"]:
                tid = obj["track_id"]
                if tid not in (video_track_id[video] + video_track_id_tree[video]):
                    continue

                # -------- STRIDE FILTER --------
                if tid in prev_frame and frame_idx - prev_frame[tid] < FRAME_STRIDE:
                    continue

                x1, y1, x2, y2 = obj["bbox"]
                crop = frame[
                    max(0, int(1.1*y1 - 0.1*y2)):min(int(1.1*y2 - 0.1*y1), frame.shape[0]),
                    max(0, int(1.1*x1 - 0.1*x2)):min(int(1.1*x2 - 0.1*x1), frame.shape[1])
                ]
                if crop.size == 0:
                    continue

                hazard_name_by_id[video].setdefault(tid, {})
                has_hazard = True

                # -------- CAPTION CACHE --------
                if tid in caption_cache:
                    txt = caption_cache[tid]
                else:
                    pv = preprocess_image(crop)
                    txt = model.chat(tokenizer, pv, PROMPT, GEN_CFG)
                    txt = clean_text(txt)
                    caption_cache[tid] = txt

                area = crop.shape[0] * crop.shape[1]
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

            # -------- TASK 3.2 (FRAME LEVEL, SPARSE) --------
            if has_hazard and frame_idx % FRAME_STRIDE == 0:
                pv = preprocess_image(frame)
                txt = model.chat(tokenizer, pv, PROMPT, GEN_CFG)
                txt = clean_text(txt)

                hazard_name_by_frame[video][txt] = (
                    hazard_name_by_frame[video].get(txt, 0.0)
                    + frame.shape[0] * frame.shape[1]
                )

        # -------- MERGE SYNONYMS --------
        for tid in hazard_name_by_id[video]:
            hazard_name_by_id[video][tid] = merge_synonyms_fast(
                hazard_name_by_id[video][tid]
            )

    except Exception as e:
        print(f"Error at {video}: {e}")
        continue

# =========================
# SAVE (UNCHANGED)
# =========================
with open("hazard_name_by_id.pkl", "wb") as f:
    pickle.dump(hazard_name_by_id, f, protocol=pickle.HIGHEST_PROTOCOL)

with open("hazard_name_by_frame.pkl", "wb") as f:
    pickle.dump(hazard_name_by_frame, f, protocol=pickle.HIGHEST_PROTOCOL)

print("✅ Task 3 FAST version finished")
