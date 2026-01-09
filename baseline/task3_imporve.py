import cv2
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
BATCHSIZE = 32        # InternVL nặng hơn BLIP2
DEVICE = "cuda"

ANNOT_PATH = "/kaggle/input/coool-dataset/annotations_public.pkl"
VIDEO_DIR = "/kaggle/input/coool-dataset/COOOL-videos"

# =========================
# LOAD DATA
# =========================
annotations = pickle.load(open(ANNOT_PATH, "rb"))
video_track_id = pickle.load(open("video_track_id.pkl", "rb"))
video_track_id_tree = pickle.load(open("video_track_id_tree.pkl", "rb"))

# =========================
# LOAD INTERNVL3.5 8B (4BIT)
# =========================
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

hazard_name_by_id = {}       # task 3.1
hazard_name_by_frame = {}    # task 3.2

SYSTEM_PROMPT = (
    "You are an autonomous driving safety system.\n"
    "Look at the image and output ONLY the main road hazard.\n"
    "Rules:\n"
    "- Use at most 3 words\n"
    "- Use a noun phrase\n"
    "- No adjectives\n"
    "- No verbs\n"
    "- No punctuation\n"
    "- If nothing dangerous, output 'none'\n"
)

def infer_batch(images):
    """
    images: list of numpy RGB images
    return: list of short hazard strings
    """
    prompts = ["<image>\n" + SYSTEM_PROMPT] * len(images)

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False
        )

    texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return [t.strip().lower() for t in texts]

# =========================
# MAIN LOOP
# =========================
for video in tqdm(sorted(annotations.keys())):
    try:
        cap = cv2.VideoCapture(f"{VIDEO_DIR}/{video}.mp4")
        num_frames = len(annotations[video])

        if video not in hazard_name_by_id:
            hazard_name_by_id[video] = {}
        if video not in hazard_name_by_frame:
            hazard_name_by_frame[video] = {}

        # buffers
        batch_ids, batch_imgs = [], []
        batch_frame_imgs = []

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
                    max(0, int(1.1 * y1 - 0.1 * y2)):
                    min(int(1.1 * y2 - 0.1 * y1), frame.shape[0]),
                    max(0, int(1.1 * x1 - 0.1 * x2)):
                    min(int(1.1 * x2 - 0.1 * x1), frame.shape[1]),
                ]

                if crop.size == 0:
                    continue

                batch_ids.append(tid)
                batch_imgs.append(crop)

                if tid not in hazard_name_by_id[video]:
                    hazard_name_by_id[video][tid] = {}

                has_hazard_in_frame = True

            if has_hazard_in_frame:
                batch_frame_imgs.append(frame)

            # ---------- TASK 3.1 ----------
            if (len(batch_imgs) >= BATCHSIZE) or (
                frame_idx == num_frames - 1 and len(batch_imgs) > 0
            ):
                texts = infer_batch(batch_imgs)

                for tid, img, txt in zip(batch_ids, batch_imgs, texts):
                    if txt == "none":
                        continue
                    if txt not in hazard_name_by_id[video][tid]:
                        hazard_name_by_id[video][tid][txt] = 0.0
                    hazard_name_by_id[video][tid][txt] += img.shape[0] * img.shape[1]

                batch_ids, batch_imgs = [], []

            # ---------- TASK 3.2 ----------
            if (len(batch_frame_imgs) >= BATCHSIZE) or (
                frame_idx == num_frames - 1 and len(batch_frame_imgs) > 0
            ):
                texts = infer_batch(batch_frame_imgs)

                for img, txt in zip(batch_frame_imgs, texts):
                    if txt == "none":
                        continue
                    if txt not in hazard_name_by_frame[video]:
                        hazard_name_by_frame[video][txt] = 0.0
                    hazard_name_by_frame[video][txt] += img.shape[0] * img.shape[1]

                batch_frame_imgs = []

    except Exception as e:
        print(f"Error at {video}: {e}")
        continue

# =========================
# SAVE (UNCHANGED FORMAT)
# =========================
with open("hazard_name_by_id_blip2opt.pkl", "wb") as f:
    pickle.dump(hazard_name_by_id, f, protocol=pickle.HIGHEST_PROTOCOL)

with open("hazard_name_by_frame_blip2opt.pkl", "wb") as f:
    pickle.dump(hazard_name_by_frame, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Task 3 done with InternVL3.5-8B (4bit)")
