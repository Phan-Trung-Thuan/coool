import cv2
import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from tqdm import tqdm
import pickle

model_path = "OpenGVLab/VideoChat-R1_5"
# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto",
)

# default processer
processor = AutoProcessor.from_pretrained(model_path)

annotations = pickle.load(open("/kaggle/input/coool-dataset/annotations_public.pkl", "rb"))
video_track_id = pickle.load(open("video_track_id.pkl", "rb"))
video_track_id_tree = pickle.load(open("video_track_id_tree.pkl", "rb"))

def draw_track_bbox_video(
    video_path,
    annotations_video,
    track_id,
    output_path,
    color=(0, 255, 0),
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
                if obj["track_id"] == track_id:
                    x1, y1, x2, y2 = map(int, obj["bbox"])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

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
            **inputs, max_new_tokens=256, use_cache=True
        )

    gen_ids = output_ids[:, inputs.input_ids.shape[1]:]
    caption = processor.batch_decode(
        gen_ids, skip_special_tokens=True
    )[0]

    return caption.strip()

hazard_caption_by_id = {}

for video in tqdm(sorted(annotations.keys())):
    try:
        video_path = f"/kaggle/input/coool-dataset/COOOL-videos/{video}.mp4"
        hazard_caption_by_id[video] = {}

        hazard_tracks = set(
            video_track_id[video] + video_track_id_tree[video]
        )

        for track_id in hazard_tracks:
            tmp_video = f"/tmp/{video}_track_{track_id}.mp4"

            draw_track_bbox_video(
                video_path,
                annotations[video],
                track_id,
                tmp_video
            )

            caption = videochat_caption(
                tmp_video,
                "Focus on the green rectangle. "
                "Describe the unusual, abnormal, or hazardous situation related "
                "to the highlighted object in this driving video.",
                model,
                processor
            )

            hazard_caption_by_id[video][track_id] = caption
            print(f"[{video} | track {track_id}] {caption}")

    except Exception as e:
        print(f"Error {video}: {e}")

with open("hazard_caption_by_id.pkl", "wb") as f:
    pickle.dump(hazard_caption_by_id, f, protocol=pickle.HIGHEST_PROTOCOL)