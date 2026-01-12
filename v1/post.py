import pickle
import pandas as pd
from tqdm import tqdm
from collections import Counter

# =========================
# LOAD DATA
# =========================

annotations = pickle.load(open("/kaggle/input/coool-dataset/annotations_public.pkl", "rb"))

# Task 1
task1 = pickle.load(open("coool\\baseline\\video_Driver_State_Changed.pkl", "rb"))

# Task 2
task2_track = pickle.load(open("coool\\baseline\\video_track_id.pkl", "rb"))
task2_tree  = pickle.load(open("coool\\baseline\\video_track_id_tree.pkl", "rb"))

# Task 3 (NEW)
task3_caption = pickle.load(open("coool\\v1\\hazard_caption_by_id.pkl", "rb"))

# =========================
# INIT OUTPUT
# =========================

df_final = pd.DataFrame()

ids = []
Driver_State_Changed = []

Hazard_Track_all = [[] for _ in range(23)]
Hazard_Name_all  = [[] for _ in range(23)]

# =========================
# MAIN LOOP
# =========================

for video in tqdm(sorted(annotations.keys())):
    try:
        ann_video = annotations[video]
        num_frames = len(ann_video)

        hazard_tracks = list(
            Counter(task2_track[video] + task2_tree[video]).keys()
        )

        # Per-video buffers
        Hazard_Track_video = [[-1]*num_frames for _ in range(23)]
        Hazard_Name_video  = [[" "] *num_frames for _ in range(23)]

        ids.extend([f"{video}_{f}" for f in range(num_frames)])

        hazard_appeared = False

        for frame_idx in range(num_frames):

            if frame_idx not in ann_video:
                for _ in range(len(hazard_tracks)):
                    Driver_State_Changed.append(False)
                continue

            frame_objs = ann_video[frame_idx]["challenge_object"]
            frame_track_ids = [o["track_id"] for o in frame_objs]

            col = 0
            for track_id in hazard_tracks:
                if col >= 23:
                    break

                if track_id in frame_track_ids:
                    hazard_appeared = True
                    Hazard_Track_video[col][frame_idx] = track_id

                    # Task 3 caption
                    if (
                        video in task3_caption
                        and track_id in task3_caption[video]
                    ):
                        Hazard_Name_video[col][frame_idx] = (
                            task3_caption[video][track_id][:100]
                        )
                    else:
                        Hazard_Name_video[col][frame_idx] = " "

                    col += 1

            # Task 1 logic (giữ nguyên semantics cũ)
            if frame_idx >= task1[video][1]:
                if not hazard_appeared and task1[video][0] == "random":
                    Driver_State_Changed.append(False)
                else:
                    Driver_State_Changed.append(True)
            else:
                Driver_State_Changed.append(False)

        for i in range(23):
            Hazard_Track_all[i].extend(Hazard_Track_video[i])
            Hazard_Name_all[i].extend(Hazard_Name_video[i])

    except Exception as e:
        print(f"[SKIP] {video} | {e}")
        continue

# =========================
# BUILD DATAFRAME
# =========================

df_final["ID"] = ids
df_final["Driver_State_Changed"] = Driver_State_Changed

for i in range(23):
    df_final[f"Hazard_Track_{i}"] = Hazard_Track_all[i]
    df_final[f"Hazard_Name_{i}"]  = Hazard_Name_all[i]

# =========================
# SAVE
# =========================

df_final.to_csv("final.csv", index=False)
print("✅ Saved final.csv")
