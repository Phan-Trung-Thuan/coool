import cv2
import torch
import pickle
import numpy as np

from tqdm import tqdm
from PIL import Image

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig

# =========================
# CONFIG
# =========================
BATCHSIZE = 128
DEVICE = "cuda"

ANNOT_PATH = "/kaggle/input/coool-dataset/annotations_public.pkl"
VIDEO_DIR  = "/kaggle/input/coool-dataset/COOOL-videos"

MODEL_NAME = "OpenGVLab/InternVL3_5-8B"

# =========================
# LOAD DATA
# =========================
annotations = pickle.load(open(ANNOT_PATH, "rb"))
video_track_id = pickle.load(open("video_track_id.pkl", "rb"))
video_track_id_tree = pickle.load(open("video_track_id_tree.pkl", "rb"))

# =========================
# LOAD INTERNVL (SAFE CONFIG)
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
# PROMPT (BLIP2-LIKE)
# =========================
PROMPT = "<image>\ncar view of"

GEN_CFG = dict(
    max_new_tokens=64,
    do_sample=False,
    temperature=0.0,
    eos_token_id=151645,
    pad_token_id=151645
)

# =========================
# IMAGE PREPROCESS (SIMPLE, FAST)
# =========================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

transform = T.Compose([
    T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def preprocess_batch(img_list):
    imgs = [transform(Image.fromarray(img)) for img in img_list]
    return torch.stack(imgs).to(torch.float16).cuda(non_blocking=True)

# =========================
# TEXT CLEAN (GIỮ NGUYÊN BLIP2)
# =========================
def clean_text(text):
    text = text.replace("car view of ", "").replace(",", "").split()
    i = 1
    while i < len(text):
        if text[i] == text[i - 1] or text[i] == "":
            text.pop(i)
        else:
            i += 1
    if len(text) > 0:
        text[0] = text[0][0].upper() + text[0][1:]
    return " ".join(text)

# =========================
# OUTPUT
# =========================
hazard_name_by_id = {}
hazard_name_by_frame = {}

# =========================
# MAIN LOOP (BLIP2 STRUCTURE)
# =========================
for video in tqdm(sorted(annotations.keys())):
    try:
        cap = cv2.VideoCapture(f"{VIDEO_DIR}/{video}.mp4")
        num_frames = len(annotations[video])

        hazard_name_by_id.setdefault(video, {})
        hazard_name_by_frame.setdefault(video, {})

        # buffers (GIỐNG BLIP2)
        batch_id, batch_img_id = [], []
        batch_img_frame = []

        for frame_idx in sorted(annotations[video].keys()):
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            has_hazard_in_frame = False

            for obj in annotations[video][frame_idx]["challenge_object"]:
                tid = obj["track_id"]
                if tid not in (video_track_id[video] + video_track_id_tree[video]):
                    continue

                x1, y1, x2, y2 = obj["bbox"]
                crop = frame[
                    max(0, int(1.1*y1 - 0.1*y2)):min(int(1.1*y2 - 0.1*y1), frame.shape[0]),
                    max(0, int(1.1*x1 - 0.1*x2)):min(int(1.1*x2 - 0.1*x1), frame.shape[1])
                ]
                if crop.size == 0:
                    continue

                batch_id.append(tid)
                batch_img_id.append(crop)

                hazard_name_by_id[video].setdefault(tid, {})
                has_hazard_in_frame = True

            if has_hazard_in_frame:
                batch_img_frame.append(frame)

            # -------- TASK 3.1 (OBJECT LEVEL) --------
            if ((len(batch_img_id) >= BATCHSIZE) or
                (frame_idx == num_frames - 1 and len(batch_img_id) > 0)):

                pv = preprocess_batch(batch_img_id)
                texts = [
                    clean_text(
                        model.chat(tokenizer, pv[i:i+1], PROMPT, GEN_CFG)
                    )
                    for i in range(len(batch_img_id))
                ]

                for tid, img, txt in zip(batch_id, batch_img_id, texts):
                    hazard_name_by_id[video][tid][txt] = (
                        hazard_name_by_id[video][tid].get(txt, 0.0)
                        + img.shape[0] * img.shape[1]
                    )

                batch_id, batch_img_id = [], []

            # -------- TASK 3.2 (FRAME LEVEL) --------
            if ((len(batch_img_frame) >= BATCHSIZE) or
                (frame_idx == num_frames - 1 and len(batch_img_frame) > 0)):

                pv = preprocess_batch(batch_img_frame)
                texts = [
                    clean_text(
                        model.chat(tokenizer, pv[i:i+1], PROMPT, GEN_CFG)
                    )
                    for i in range(len(batch_img_frame))
                ]

                for img, txt in zip(batch_img_frame, texts):
                    hazard_name_by_frame[video][txt] = (
                        hazard_name_by_frame[video].get(txt, 0.0)
                        + img.shape[0] * img.shape[1]
                    )

                batch_img_frame = []

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

print("✅ Task 3 InternVL (BLIP2-structure) finished")
