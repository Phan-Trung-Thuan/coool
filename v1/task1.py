# ============================================================
# COOOL Task 1 - Driver State Change Detection (Clean & Fast)
# Ego-motion + Hyperparameter tuning + Change-point
# ============================================================

import numpy as np
import pickle
from tqdm import tqdm
from moviepy.editor import VideoFileClip
from scipy.signal import savgol_filter
import ruptures as rpt
import cv2

# ---------------- CONFIG ----------------
IMG_W, IMG_H = 1280, 720

ALPHA_LIST = [0.8, 1.0, 1.2]   # motion weight
BETA_LIST  = [0.4, 0.6, 0.8]   # sound weight
PEN_LIST   = [3, 5, 8]         # change-point penalty

REACTION_DELAY = 10            # frames (~300–400ms)
SMOOTH_WIN = 9

# ----------------------------------------


# ================= UTILS =================
def bbox_center(b):
    x1, y1, x2, y2 = b
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

def bbox_area(b):
    x1, y1, x2, y2 = b
    return abs(x2 - x1) * abs(y2 - y1)

def temporal_smooth(x, win=9):
    if len(x) < win:
        return x
    return savgol_filter(x, win, polyorder=2)

def normalize(x):
    return x / (x.max() + 1e-6)

# ========================================


# ============ EGO MOTION =================
def estimate_ego_motion(video_path, num_frames, step=3):
    """
    Estimate global camera motion using optical flow (sparse sampling)
    """
    cap = cv2.VideoCapture(video_path)
    ret, prev = cap.read()
    if not ret:
        cap.release()
        return np.zeros((num_frames, 2))

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    motions = [np.zeros(2)]

    for i in range(1, num_frames):
        ret, frame = cap.read()
        if not ret:
            motions.append(motions[-1])
            continue

        if i % step != 0:
            motions.append(motions[-1])
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        mean_flow = flow.mean(axis=(0, 1))
        motions.append(mean_flow)
        prev_gray = gray

    cap.release()
    return np.array(motions)

# ========================================


# ============ MOTION ENERGY ==============
def compute_motion_energy(annotations, video, ego_motion):
    num_frames = len(annotations[video])
    energy = np.zeros(num_frames)
    prev_centers = {}

    for f in range(num_frames):
        frame = annotations[video][f]
        ego = ego_motion[f]

        for group in ["traffic_scene", "challenge_object"]:
            for obj in frame[group]:
                tid = obj["track_id"]
                bbox = obj["bbox"]
                center = bbox_center(bbox)
                area = bbox_area(bbox)

                # importance weight
                size_w = area / (IMG_W * IMG_H)
                center_w = 1 - np.linalg.norm(
                    center - np.array([IMG_W/2, IMG_H/2])
                ) / np.linalg.norm([IMG_W/2, IMG_H/2])

                weight = max(0, size_w * center_w)

                if tid in prev_centers:
                    raw_motion = center - prev_centers[tid]
                    compensated = raw_motion - ego
                    velocity = np.linalg.norm(compensated)
                    energy[f] += weight * velocity

                prev_centers[tid] = center

    return temporal_smooth(energy, SMOOTH_WIN)

# ========================================


# ============ SOUND ENERGY ================
def compute_sound_energy(video_path, num_frames, fps_audio=60):
    clip = VideoFileClip(video_path)
    audio_energy = []

    for frame in clip.audio.iter_frames(fps=fps_audio):
        audio_energy.append(np.mean(np.abs(frame)))

    audio_energy = np.array(audio_energy)
    audio_energy = (audio_energy - audio_energy.mean()) / (audio_energy.std() + 1e-6)

    idx = np.linspace(0, len(audio_energy)-1, num_frames).astype(int)
    return temporal_smooth(audio_energy[idx], SMOOTH_WIN)

# ========================================


# ============ CHANGE POINT ================
def detect_change_point(signal, pen):
    if len(signal) < 20:
        return len(signal) // 2

    algo = rpt.Pelt(model="rbf").fit(signal)
    cps = algo.predict(pen=pen)
    return cps[0] if len(cps) > 0 else len(signal) // 2

# ========================================


# ================= MAIN ===================
annotations = pickle.load(open("/kaggle/input/annotations_public.pkl", "rb"))
video_Driver_State_Changed = {}

for video in tqdm(sorted(annotations.keys())):
    num_frames = len(annotations[video])
    video_path = f"/kaggle/input/COOOL-videos/{video}.mp4"

    # Ego motion
    try:
        ego_motion = estimate_ego_motion(video_path, num_frames)
    except:
        ego_motion = np.zeros((num_frames, 2))

    # Signals
    motion_energy = compute_motion_energy(annotations, video, ego_motion)

    try:
        sound_energy = compute_sound_energy(video_path, num_frames)
    except:
        sound_energy = np.zeros(num_frames)

    motion_energy = normalize(motion_energy)
    sound_energy  = normalize(sound_energy)

    best_score = -1
    best_cp = num_frames // 2
    best_cfg = None

    # Hyperparameter tuning (light)
    for alpha in ALPHA_LIST:
        for beta in BETA_LIST:
            fused = temporal_smooth(alpha * motion_energy + beta * sound_energy)
            for pen in PEN_LIST:
                cp = detect_change_point(fused, pen)
                score = fused[cp:].mean() - fused[:cp].mean()

                if score > best_score:
                    best_score = score
                    best_cp = cp
                    best_cfg = (alpha, beta, pen)

    reaction_frame = min(best_cp + REACTION_DELAY, num_frames - 1)

    video_Driver_State_Changed[video] = {
        "change_frame": reaction_frame,
        "cp": best_cp,
        "alpha": best_cfg[0],
        "beta": best_cfg[1],
        "pen": best_cfg[2]
    }

# Save
with open("video_Driver_State_Changed.pkl", "wb") as f:
    pickle.dump(video_Driver_State_Changed, f)