import cv2
from sympy import true
import torch
import pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

# =========================
# CONFIG
# =========================
BATCHSIZE = 64
DEVICE = "cuda"

ANNOT_PATH = "/kaggle/input/coool-dataset/annotations_public.pkl"
VIDEO_DIR = "/kaggle/input/coool-dataset/COOOL-videos"

annotations = pickle.load(open(ANNOT_PATH, "rb"))
video_track_id = pickle.load(open("video_track_id.pkl", "rb"))
video_track_id_tree = pickle.load(open("video_track_id_tree.pkl", "rb"))

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

MODEL_NAME = "OpenGVLab/InternVL3_5-8B"

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
# OUTPUT
# =========================
hazard_name_by_id = {}
hazard_name_by_frame = {}

# =========================
# PROMPT (BLIP2-like)
# =========================
PROMPT = (
    "Describe the scene from a car camera. "
    "Focus on the main object that could be dangerous. "
    "Use a short sentence describing the object and its action or position."
)

def infer_batch(images):
    prompts = ["<image>\n" + PROMPT] * len(images)
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False
        )

    texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return [t.strip() for t in texts]

# =========================
# MAIN LOOP
# =========================
for video in tqdm(sorted(annotations.keys())):
    # try:
    if True:
        cap = cv2.VideoCapture(f"{VIDEO_DIR}/{video}.mp4")
        num_frames = len(annotations[video])

        hazard_name_by_id.setdefault(video, {})
        hazard_name_by_frame.setdefault(video, {})

        batch_ids, batch_imgs, batch_areas = [], [], []
        batch_frame_imgs = []

        prev_area = defaultdict(lambda: None)

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

                x1, y1, x2, y2 = obj["bbox"]
                crop = frame[
                    max(0, int(1.1*y1 - 0.1*y2)):min(int(1.1*y2 - 0.1*y1), frame.shape[0]),
                    max(0, int(1.1*x1 - 0.1*x2)):min(int(1.1*x2 - 0.1*x1), frame.shape[1])
                ]

                if crop.size == 0:
                    continue

                area = crop.shape[0] * crop.shape[1]

                batch_ids.append(tid)
                batch_imgs.append(crop)
                batch_areas.append(area)

                hazard_name_by_id[video].setdefault(tid, {})
                has_hazard = True

            if has_hazard:
                batch_frame_imgs.append(frame)

            # -------- TASK 3.1 --------
            if len(batch_imgs) >= BATCHSIZE or (frame_idx == num_frames - 1 and batch_imgs):
                texts = infer_batch(batch_imgs)

                for tid, img, area, txt in zip(batch_ids, batch_imgs, batch_areas, texts):
                    # ----- TTC / Ego-motion -----
                    prev = prev_area[tid]
                    if prev is None:
                        motion_weight = 1.0
                    else:
                        growth = max((area - prev) / (prev + 1e-6), 0)
                        ttc = 1.0 / (growth + 1e-3)
                        motion_weight = np.exp(-ttc / 6.0)

                    prev_area[tid] = area

                    score = area * motion_weight
                    hazard_name_by_id[video][tid][txt] = (
                        hazard_name_by_id[video][tid].get(txt, 0.0) + score
                    )

                batch_ids, batch_imgs, batch_areas = [], [], []

            # -------- TASK 3.2 --------
            if len(batch_frame_imgs) >= BATCHSIZE or (frame_idx == num_frames - 1 and batch_frame_imgs):
                texts = infer_batch(batch_frame_imgs)

                for img, txt in zip(batch_frame_imgs, texts):
                    hazard_name_by_frame[video][txt] = (
                        hazard_name_by_frame[video].get(txt, 0.0)
                        + img.shape[0] * img.shape[1]
                    )

                batch_frame_imgs = []

    # except Exception as e:
    #     print(f"Error at {video}: {e}")
    #     continue

# =========================
# SAVE (UNCHANGED)
# =========================
with open("hazard_name_by_id.pkl", "wb") as f:
    pickle.dump(hazard_name_by_id, f, protocol=pickle.HIGHEST_PROTOCOL)

with open("hazard_name_by_frame.pkl", "wb") as f:
    pickle.dump(hazard_name_by_frame, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Task 3 done with InternVL 3.5 8B + TTC / ego-motion")
