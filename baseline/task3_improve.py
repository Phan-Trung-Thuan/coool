import cv2
import torch
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

# ================= CONFIG =================
DEVICE = "cuda"
DTYPE = torch.bfloat16
MODEL_PATH = "OpenGVLab/InternVL3_5-8B"

# ================= LOAD DATA =================
annotations = pickle.load(open("/kaggle/input/coool-dataset/annotations_public.pkl", "rb"))
video_track_id = pickle.load(open("video_track_id.pkl", "rb"))
video_track_id_tree = pickle.load(open("video_track_id_tree.pkl", "rb"))

# ================= INTERNVL IMAGE PREPROCESS =================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size=448):
    return T.Compose([
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

def dynamic_preprocess(image, image_size=448, max_num=12, use_thumbnail=True):
    orig_w, orig_h = image.size
    aspect_ratio = orig_w / orig_h

    target_ratios = sorted(
        [(i, j) for n in range(1, max_num + 1)
         for i in range(1, n + 1)
         for j in range(1, n + 1)
         if i * j <= max_num],
        key=lambda x: x[0] * x[1]
    )

    best_ratio = min(
        target_ratios,
        key=lambda r: abs(aspect_ratio - r[0] / r[1])
    )

    target_w = image_size * best_ratio[0]
    target_h = image_size * best_ratio[1]
    resized = image.resize((target_w, target_h))

    patches = []
    for i in range(best_ratio[0] * best_ratio[1]):
        x = (i % best_ratio[0]) * image_size
        y = (i // best_ratio[0]) * image_size
        patches.append(resized.crop((x, y, x + image_size, y + image_size)))

    if use_thumbnail and len(patches) > 1:
        patches.append(image.resize((image_size, image_size)))

    return patches

def image_to_pixel_values(img_np, max_num=12):
    image = Image.fromarray(img_np).convert("RGB")
    transform = build_transform(448)
    patches = dynamic_preprocess(image, max_num=max_num)
    pixel_values = torch.stack([transform(p) for p in patches])
    return pixel_values

# ================= LOAD INTERNVL MODEL =================
model = AutoModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=DTYPE,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    device_map="auto"
).eval()

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    use_fast=False
)

# ================= TEXT CLEAN =================
def clean_text(text):
    text = text.replace(",", " ").replace(".", " ")
    words = text.split()
    out = []
    for w in words:
        if len(out) == 0 or w != out[-1]:
            out.append(w)
    if len(out) > 0:
        out[0] = out[0].capitalize()
    return " ".join(out)

# ================= INTERNVL CAPTION =================
def internvl_caption(img_np, prompt):
    pixel_values = image_to_pixel_values(img_np).to(DTYPE).to(DEVICE)
    question = "<image>\n" + prompt

    with torch.no_grad():
        response = model.chat(
            tokenizer,
            pixel_values,
            question,
            generation_config=dict(
                max_new_tokens=64,
                do_sample=False,
                eos_token_id=151645,
                pad_token_id=151645,
            )
        )
    return clean_text(response)

# ================= OUTPUT DICTS =================
hazard_name_by_id = {}
hazard_name_by_frame = {}

# ================= MAIN LOOP =================
for video in tqdm(sorted(annotations.keys())[:2]):
    try:
        cap = cv2.VideoCapture(
            f"/kaggle/input/coool-dataset/COOOL-videos/{video}.mp4"
        )

        hazard_name_by_id.setdefault(video, {})
        hazard_name_by_frame.setdefault(video, {})

        for frame_idx in sorted(annotations[video].keys()):
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            has_hazard = False

            # ---------- TASK 3.1: OBJECT LEVEL ----------
            for obj in annotations[video][frame_idx]["challenge_object"]:
                tid = obj["track_id"]
                if tid not in (video_track_id[video] + video_track_id_tree[video]):
                    continue

                x1, y1, x2, y2 = obj["bbox"]
                crop = frame[
                    max(0, int(1.1*y1 - 0.1*y2)):min(int(1.1*y2 - 0.1*y1), frame.shape[0]),
                    max(0, int(1.1*x1 - 0.1*x2)):min(int(1.1*x2 - 0.1*x1), frame.shape[1]),
                ]

                txt = internvl_caption(
                    crop,
                    "Identify and name the traffic hazard object that may pose a risk to the vehicle in short phrases."
                )

                hazard_name_by_id[video].setdefault(tid, {})
                hazard_name_by_id[video][tid][txt] = \
                    hazard_name_by_id[video][tid].get(txt, 0.0) + crop.shape[0] * crop.shape[1]

                has_hazard = True

            # ---------- TASK 3.2: FRAME LEVEL ----------
            if has_hazard:
                txt = internvl_caption(
                    frame,
                    "Describe the main traffic hazard visible in this driving scene in short phrases."
                )
                hazard_name_by_frame[video][txt] = \
                    hazard_name_by_frame[video].get(txt, 0.0) + frame.shape[0] * frame.shape[1]

        cap.release()

    except Exception as e:
        print(f"[ERROR] {video}: {e}")
        continue

# ================= SAVE =================
with open("hazard_name_by_id.pkl", "wb") as f:
    pickle.dump(hazard_name_by_id, f, protocol=pickle.HIGHEST_PROTOCOL)

with open("hazard_name_by_frame.pkl", "wb") as f:
    pickle.dump(hazard_name_by_frame, f, protocol=pickle.HIGHEST_PROTOCOL)
