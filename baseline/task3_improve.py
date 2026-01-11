import cv2
import os
import torch
import pickle
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# =========================================================
# CONFIG
# =========================================================
DEVICE = "cuda"
MODEL_PATH = "OpenGVLab/VideoChat-R1_5"
TMP_DIR = "/tmp"

os.makedirs(TMP_DIR, exist_ok=True)

# =========================================================
# LOAD MODEL
# =========================================================
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="auto"
)
model.eval()

processor = AutoProcessor.from_pretrained(MODEL_PATH)

annotations = pickle.load(open(
    "/kaggle/input/coool-dataset/annotations_public.pkl", "rb"
))
video_track_id = pickle.load(open("video_track_id.pkl", "rb"))
video_track_id_tree = pickle.load(open("video_track_id_tree.pkl", "rb"))

COLOR_TABLE = [
    ("green",   (0, 255, 0)),
    ("red",     (255, 0, 0)),
    ("blue",    (0, 0, 255)),
    ("yellow",  (255, 255, 0)),
    ("cyan",    (0, 255, 255)),
    ("magenta", (255, 0, 255)),
]

def assign_colors_to_tracks(track_ids):
    mapping = {}
    for i, tid in enumerate(sorted(track_ids)):
        cname, cbgr = COLOR_TABLE[i % len(COLOR_TABLE)]
        mapping[tid] = {
            "color_name": cname,
            "color_bgr": cbgr
        }
    return mapping

def draw_multi_track_bbox_video(
    video_path,
    annotations_video,
    track_color_map,
    output_path,
    thickness=3
):
    cap = cv2.VideoCapture(video_path)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in annotations_video:
            for obj in annotations_video[frame_idx]["challenge_object"]:
                tid = obj["track_id"]
                if tid in track_color_map:
                    x1, y1, x2, y2 = map(int, obj["bbox"])
                    color = track_color_map[tid]["color_bgr"]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

def build_color_prompt(track_color_map):
    lines = [
        "The video contains multiple highlighted objects, each marked by a different colored rectangle.",
        "For each colored rectangle, describe the unusual, abnormal, or hazardous situation related to that object in short phrases.",
        "Return the results strictly in the following format:",
        "<color>: <hazard description>",
        "",
        "Color mapping:"
    ]

    for tid, info in track_color_map.items():
        lines.append(f"- {info['color_name']} rectangle corresponds to one object")

    return "\n".join(lines)

# =========================================================
# VIDEOCHAT INFERENCE
# =========================================================
def videochat_caption(video_path, prompt, model, processor, device="cuda"):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "total_pixels": 128 * 12 * 28 * 28,
                    "min_pixels": 128 * 28 * 28,
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True
    )

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        fps=video_kwargs["fps"],
        padding=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            use_cache=True
        )

    gen_ids = output_ids[:, inputs.input_ids.shape[1]:]
    output_text = processor.batch_decode(
        gen_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )[0]

    return output_text.strip()

# =========================================================
# PARSE COLOR-BASED OUTPUT
# =========================================================
def parse_color_captions(text, track_color_map):
    color_to_tid = {
        info["color_name"]: tid
        for tid, info in track_color_map.items()
    }

    result = {}
    for line in text.split("\n"):
        if ":" not in line:
            continue
        color, desc = line.split(":", 1)
        color = color.strip().lower()
        if color in color_to_tid:
            result[color_to_tid[color]] = desc.strip()

    return result

# =========================================================
# MAIN LOOP — TASK 3.1
# =========================================================
hazard_caption_by_id = {}

for video in tqdm(sorted(annotations.keys())):
    try:
        video_path = f"/kaggle/input/coool-dataset/COOOL-videos/{video}.mp4"
        hazard_caption_by_id[video] = {}

        hazard_tracks = set(
            video_track_id[video] + video_track_id_tree[video]
        )
        if len(hazard_tracks) == 0:
            continue

        track_color_map = assign_colors_to_tracks(hazard_tracks)

        tmp_video = os.path.join(TMP_DIR, f"{video}_multi_track.mp4")

        draw_multi_track_bbox_video(
            video_path,
            annotations[video],
            track_color_map,
            tmp_video
        )

        prompt = build_color_prompt(track_color_map)

        raw_output = videochat_caption(
            tmp_video,
            prompt,
            model,
            processor,
            device=DEVICE
        )

        parsed = parse_color_captions(raw_output, track_color_map)
        hazard_caption_by_id[video].update(parsed)

        print(f"\n[{video}]")
        for tid, cap in parsed.items():
            print(f"  track {tid}: {cap}")

    except Exception as e:
        print(f"Error at {video}: {e}")

# =========================================================
# SAVE RESULT
# =========================================================
with open("hazard_caption_by_id.pkl", "wb") as f:
    pickle.dump(
        hazard_caption_by_id,
        f,
        protocol=pickle.HIGHEST_PROTOCOL
    )
