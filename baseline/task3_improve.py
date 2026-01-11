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

# ============================================================
# CONFIG
# ============================================================

ANNOT_PATH = "/kaggle/input/coool-dataset/annotations_public.pkl"
VIDEO_DIR = "/kaggle/input/coool-dataset/COOOL-videos"

MODEL_NAME = "OpenGVLab/InternVL3_5-8B"
DEVICE = "cuda"

FRAME_STRIDE = 5          # caption mỗi track mỗi N frame
MAX_NEW_TOKENS = 20       # ép output ngắn như BLIP2

# ============================================================
# LOAD DATA
# ============================================================

annotations = pickle.load(open(ANNOT_PATH, "rb"))
video_track_id = pickle.load(open("video_track_id.pkl", "rb"))
video_track_id_tree = pickle.load(open("video_track_id_tree.pkl", "rb"))

# ============================================================
# LOAD INTERNVL (STABLE CONFIG)
# ============================================================

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

# ============================================================
# BLIP2-STYLE PROMPT  (RẤT QUAN TRỌNG)
# ============================================================

PROMPT = (
    "<image>\n"
    "Describe the single main dangerous object using ONE short noun phrase only. "
    "Examples: 'A dog on the road', 'Moose crossing road', 'A fallen tree'. "
    "Do NOT describe the scene. Do NOT use full sentences."
)

GEN_CFG = dict(
    max_new_tokens=MAX_NEW_TOKENS,
    do_sample=False,
    temperature=0.0,
    eos_token_id=151645,
    pad_token_id=151645
)

# ============================================================
# IMAGE PREPROCESS (FAST)
# ============================================================

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

transform = T.Compose([
    T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def preprocess_image(np_img):
    img = Image.fromarray(np_img)
    return transform(img).unsqueeze(0).to(
        torch.float16
    ).cuda(non_blocking=True)

# ============================================================
# TEXT NORMALIZATION (MATCH BLIP2 DISTRIBUTION)
# ============================================================

def normalize_caption(txt: str) -> str:
    if not txt:
        return ""

    # remove line breaks
    txt = txt.split("\n")[0]

    # remove common InternVL prefixes
    for p in [
        "The image shows",
        "This image shows",
        "Here are",
        "In the image",
        "The road is",
    ]:
        if p.lower() in txt.lower():
            txt = txt.split(".")[0]

    # remove bullet symbols
    txt = txt.replace("-", "").strip()

    # only first sentence
    txt = txt.split(".")[0]

    # capitalize
    if len(txt) > 0:
        txt = txt[0].upper() + txt[1:]

    return txt.strip()

# ============================================================
# OUTPUT STRUCTURES (UNCHANGED)
# ============================================================

hazard_name_by_id = {}
hazard_name_by_frame = {}

# ============================================================
# MAIN LOOP
# ============================================================

for video in tqdm(sorted(annotations.keys())[:2]):
    try:
        cap = cv2.VideoCapture(f"{VIDEO_DIR}/{video}.mp4")
        num_frames = len(annotations[video])

        hazard_name_by_id.setdefault(video, {})
        hazard_name_by_frame.setdefault(video, {})

        last_caption_frame = {}
        caption_cache = {}

        for frame_idx in sorted(annotations[video].keys()):
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            has_hazard = False

            # ===============================
            # TASK 3.1 — OBJECT LEVEL
            # ===============================
            for obj in annotations[video][frame_idx]["challenge_object"]:
                tid = obj["track_id"]

                if tid not in (video_track_id[video] + video_track_id_tree[video]):
                    continue

                # stride filter per track
                if tid in last_caption_frame:
                    if frame_idx - last_caption_frame[tid] < FRAME_STRIDE:
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

                # caption cache per track
                if tid in caption_cache:
                    txt = caption_cache[tid]
                else:
                    pv = preprocess_image(crop)
                    txt = model.chat(tokenizer, pv, PROMPT, GEN_CFG)
                    txt = normalize_caption(txt)
                    caption_cache[tid] = txt

                area = crop.shape[0] * crop.shape[1]

                hazard_name_by_id[video][tid][txt] = (
                    hazard_name_by_id[video][tid].get(txt, 0.0) + area
                )

                last_caption_frame[tid] = frame_idx

            # ===============================
            # TASK 3.2 — FRAME LEVEL
            # (KHÔNG STRIDE CỨNG → tránh miss video_0001)
            # ===============================
            if has_hazard:
                pv = preprocess_image(frame)
                txt = model.chat(tokenizer, pv, PROMPT, GEN_CFG)
                txt = normalize_caption(txt)

                hazard_name_by_frame[video][txt] = (
                    hazard_name_by_frame[video].get(txt, 0.0)
                    + frame.shape[0] * frame.shape[1]
                )

    except Exception as e:
        print(f"[ERROR] {video}: {e}")
        continue

# ============================================================
# SAVE (UNCHANGED)
# ============================================================
print(hazard_name_by_id)
print(hazard_name_by_frame)

with open("hazard_name_by_id.pkl", "wb") as f:
    pickle.dump(hazard_name_by_id, f, protocol=pickle.HIGHEST_PROTOCOL)

with open("hazard_name_by_frame.pkl", "wb") as f:
    pickle.dump(hazard_name_by_frame, f, protocol=pickle.HIGHEST_PROTOCOL)

print("✅ InternVL BLIP2-style Task 3 finished")