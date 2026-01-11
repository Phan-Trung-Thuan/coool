import cv2
import torch
import pickle
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

# ================= CONFIG =================
BATCHSIZE = 32        # InternVL nặng hơn BLIP2
DEVICE = "cuda"
DTYPE = torch.float16
MODEL_NAME = "OpenGVLab/InternVL3_5-8B"

# ================= LOAD DATA =================
annotations = pickle.load(open("/kaggle/input/coool-dataset/annotations_public.pkl", "rb"))
video_track_id = pickle.load(open("video_track_id.pkl", "rb"))
video_track_id_tree = pickle.load(open("video_track_id_tree.pkl", "rb"))

# ================= LOAD MODEL =================
processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,
    device_map="auto",
    trust_remote_code=True
).eval()

# ================= OUTPUT DICTS =================
hazard_name_by_id = {}
hazard_name_by_frame = {}

# ================= UTILS =================
def clean_text(text):
    text = text.replace(",", " ").replace(".", " ")
    words = text.split()
    dedup = []
    for w in words:
        if len(dedup) == 0 or w != dedup[-1]:
            dedup.append(w)
    if len(dedup) > 0:
        dedup[0] = dedup[0].capitalize()
    return " ".join(dedup)

def internvl_caption(images, prompt):
    """
    images: list of np.ndarray RGB
    """
    conversations = []
    for _ in images:
        conversations.append([
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ])

    inputs = processor(
        images=images,
        conversations=conversations,
        return_tensors="pt"
    ).to(DEVICE, DTYPE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False
        )

    texts = processor.batch_decode(outputs, skip_special_tokens=True)
    return [clean_text(t) for t in texts]

# ================= MAIN LOOP =================
for video in tqdm(sorted(annotations.keys())[:2]):
    try:
        cap = cv2.VideoCapture(
            f"/kaggle/input/coool-dataset/COOOL-videos/{video}.mp4"
        )

        if video not in hazard_name_by_id:
            hazard_name_by_id[video] = {}
        if video not in hazard_name_by_frame:
            hazard_name_by_frame[video] = {}

        batch_ids, batch_crops = [], []
        batch_frames = []

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
                    max(0, int(1.1*x1 - 0.1*x2)):min(int(1.1*x2 - 0.1*x1), frame.shape[1]),
                ]

                batch_ids.append(tid)
                batch_crops.append(crop)
                has_hazard = True

                if tid not in hazard_name_by_id[video]:
                    hazard_name_by_id[video][tid] = {}

            if has_hazard:
                batch_frames.append(frame)

            # -------- TASK 3.1 (Object-level) --------
            if len(batch_crops) >= BATCHSIZE:
                captions = internvl_caption(
                    batch_crops,
                    "Describe the hazard object in front of the vehicle in short phrases."
                )

                for tid, img, txt in zip(batch_ids, batch_crops, captions):
                    hazard_name_by_id[video][tid][txt] = \
                        hazard_name_by_id[video][tid].get(txt, 0.0) + img.shape[0]*img.shape[1]

                batch_ids, batch_crops = [], []

            # -------- TASK 3.2 (Frame-level) --------
            if len(batch_frames) >= BATCHSIZE:
                captions = internvl_caption(
                    batch_frames,
                    "Describe the main traffic hazard visible in this driving scene in short phrases."
                )

                for img, txt in zip(batch_frames, captions):
                    hazard_name_by_frame[video][txt] = \
                        hazard_name_by_frame[video].get(txt, 0.0) + img.shape[0]*img.shape[1]

                batch_frames = []

        cap.release()

    except Exception as e:
        print(f"[ERROR] {video}: {e}")
        continue

# ================= SAVE =================
with open("hazard_name_by_id.pkl", "wb") as f:
    pickle.dump(hazard_name_by_id, f, protocol=pickle.HIGHEST_PROTOCOL)

with open("hazard_name_by_frame.pkl", "wb") as f:
    pickle.dump(hazard_name_by_frame, f, protocol=pickle.HIGHEST_PROTOCOL)
