"""
Exam Proctoring App (GUI)
===========================
A desktop application that ties together the three pieces built for this
project into one tool:

    1. Face detection      (Haar cascade)
    2. Head pose estimation (MediaPipe Face Mesh + solvePnP)
    3. Alert system         (no face / multiple faces / looking away /
                              camera tampering), logged to CSV with a
                              live trust score.

The GUI is built with Tkinter (no extra install needed) and shows the
live annotated video feed alongside a running alert panel.

Dependencies:
    pip install opencv-python mediapipe numpy pillow

Usage:
    python proctoring_app.py
"""

import cv2
import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import ttk, scrolledtext
from PIL import Image, ImageTk
import csv
import os
import time
from datetime import datetime
from collections import deque


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
class Config:
    NO_FACE_ALERT_SECONDS = 3
    LOOK_AWAY_ALERT_SECONDS = 3
    MULTI_FACE_COOLDOWN = 5
    NO_FACE_COOLDOWN = 5
    LOOK_AWAY_COOLDOWN = 5
    TAMPER_COOLDOWN = 5
    MIN_FACE_AREA_RATIO = 0.02
    MAX_FACE_AREA_RATIO = 0.55
    YAW_THRESHOLD = 15.0
    PITCH_THRESHOLD = 12.0
    LOG_FILE = "proctoring_log.csv"

    WEIGHTS = {
        "NO_FACE": 5,
        "MULTIPLE_FACES": 10,
        "LOOKING_AWAY": 3,
        "CAMERA_TAMPER": 8,
    }


MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0),
], dtype=np.float64)

LANDMARK_IDS = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_left_corner": 33,
    "right_eye_right_corner": 263,
    "left_mouth_corner": 61,
    "right_mouth_corner": 291,
}


# --------------------------------------------------------------------------- #
# Alert Logger
# --------------------------------------------------------------------------- #
class AlertLogger:
    def __init__(self, log_file, ui_callback):
        self.log_file = log_file
        self.counts = {}
        self.ui_callback = ui_callback
        is_new = not os.path.exists(log_file)
        self._file = open(log_file, "a", newline="")
        self._writer = csv.writer(self._file)
        if is_new:
            self._writer.writerow(["timestamp", "alert_type", "message"])

    def raise_alert(self, alert_type, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._writer.writerow([timestamp, alert_type, message])
        self._file.flush()
        self.counts[alert_type] = self.counts.get(alert_type, 0) + 1
        self.ui_callback(f"[{timestamp}] {alert_type}: {message}")

    def trust_score(self):
        penalty = sum(self.counts.get(k, 0) * w for k, w in Config.WEIGHTS.items())
        return max(0, 100 - penalty)

    def close(self):
        self._file.close()


# --------------------------------------------------------------------------- #
# Core proctoring logic (face detection + head pose + alert rules)
# --------------------------------------------------------------------------- #
class ProctoringEngine:
    def __init__(self, logger):
        self.logger = logger
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )

        self.no_face_since = None
        self.look_away_since = None
        self.last_multi_face_alert = 0
        self.last_no_face_alert = 0
        self.last_look_away_alert = 0
        self.last_tamper_alert = 0
        self.face_count_buffer = deque(maxlen=5)

    def _get_image_points(self, landmarks, w, h):
        pts = []
        for name in ["nose_tip", "chin", "left_eye_left_corner",
                     "right_eye_right_corner", "left_mouth_corner", "right_mouth_corner"]:
            lm = landmarks[LANDMARK_IDS[name]]
            pts.append((lm.x * w, lm.y * h))
        return np.array(pts, dtype=np.float64)

    def _estimate_pose(self, image_points, w, h):
        focal_length = w
        camera_matrix = np.array([
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))
        ok, rot_vec, trans_vec = cv2.solvePnP(
            MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None, None
        rot_mat, _ = cv2.Rodrigues(rot_vec)
        pose_mat = cv2.hconcat((rot_mat, trans_vec))
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(pose_mat)
        pitch, yaw, _ = euler.flatten()
        pitch = pitch - 180 if pitch > 90 else (pitch + 180 if pitch < -90 else pitch)
        return pitch, yaw

    def _check_face_count(self, faces, w, h):
        now = time.time()
        self.face_count_buffer.append(len(faces))
        stable_count = max(set(self.face_count_buffer), key=self.face_count_buffer.count)

        if stable_count == 0:
            if self.no_face_since is None:
                self.no_face_since = now
            elapsed = now - self.no_face_since
            if elapsed >= Config.NO_FACE_ALERT_SECONDS and \
               now - self.last_no_face_alert > Config.NO_FACE_COOLDOWN:
                self.logger.raise_alert("NO_FACE", f"No face detected for {elapsed:.1f}s.")
                self.last_no_face_alert = now
        else:
            self.no_face_since = None

        if stable_count > 1 and now - self.last_multi_face_alert > Config.MULTI_FACE_COOLDOWN:
            self.logger.raise_alert("MULTIPLE_FACES", f"{stable_count} faces detected.")
            self.last_multi_face_alert = now

        if stable_count == 1:
            x, y, fw, fh = faces[0]
            ratio = (fw * fh) / (w * h)
            now2 = time.time()
            if (ratio < Config.MIN_FACE_AREA_RATIO or ratio > Config.MAX_FACE_AREA_RATIO) and \
               now2 - self.last_tamper_alert > Config.TAMPER_COOLDOWN:
                self.logger.raise_alert("CAMERA_TAMPER", f"Unusual face size ratio ({ratio:.2f}).")
                self.last_tamper_alert = now2

        return stable_count

    def _check_gaze(self, pitch, yaw):
        now = time.time()
        off_center = abs(yaw) > Config.YAW_THRESHOLD or abs(pitch) > Config.PITCH_THRESHOLD
        if off_center:
            if self.look_away_since is None:
                self.look_away_since = now
            elapsed = now - self.look_away_since
            if elapsed >= Config.LOOK_AWAY_ALERT_SECONDS and \
               now - self.last_look_away_alert > Config.LOOK_AWAY_COOLDOWN:
                self.logger.raise_alert("LOOKING_AWAY", f"Off-center for {elapsed:.1f}s.")
                self.last_look_away_alert = now
        else:
            self.look_away_since = None

    def process_frame(self, frame):
        """Runs detection + pose + alert rules on one frame, returns annotated frame."""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60))

        stable_count = self._check_face_count(faces, w, h)
        for (x, y, fw, fh) in faces:
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)

        direction = "N/A"
        if stable_count == 1:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_face_mesh.process(rgb)
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                image_points = self._get_image_points(landmarks, w, h)
                pitch, yaw = self._estimate_pose(image_points, w, h)
                if pitch is not None:
                    self._check_gaze(pitch, yaw)
                    if abs(yaw) > Config.YAW_THRESHOLD:
                        direction = "Looking Right" if yaw > 0 else "Looking Left"
                    elif abs(pitch) > Config.PITCH_THRESHOLD:
                        direction = "Looking Down" if pitch > 0 else "Looking Up"
                    else:
                        direction = "Center"

        cv2.putText(frame, f"Faces: {stable_count} | Gaze: {direction}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        return frame

    def close(self):
        self.mp_face_mesh.close()


# --------------------------------------------------------------------------- #
# GUI Application
# --------------------------------------------------------------------------- #
class ProctoringApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Exam Proctoring App")
        self.root.geometry("980x600")
        self.root.resizable(False, False)

        self.running = False
        self.cap = None
        self.logger = AlertLogger(Config.LOG_FILE, self._log_to_ui)
        self.engine = ProctoringEngine(self.logger)

        self._build_ui()

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)

        # Video panel
        video_frame = ttk.LabelFrame(main, text="Live Feed", padding=5)
        video_frame.grid(row=0, column=0, rowspan=2, sticky="n")
        self.video_label = ttk.Label(video_frame)
        self.video_label.pack()

        # Controls
        controls = ttk.Frame(main, padding=5)
        controls.grid(row=0, column=1, sticky="n")

        self.start_btn = ttk.Button(controls, text="Start Session", command=self.start)
        self.start_btn.pack(fill="x", pady=3)
        self.stop_btn = ttk.Button(controls, text="Stop Session", command=self.stop, state="disabled")
        self.stop_btn.pack(fill="x", pady=3)

        self.trust_var = tk.StringVar(value="Trust Score: 100/100")
        ttk.Label(controls, textvariable=self.trust_var, font=("Segoe UI", 12, "bold")).pack(pady=10)

        # Alert log panel
        log_frame = ttk.LabelFrame(main, text="Alert Log", padding=5)
        log_frame.grid(row=1, column=1, sticky="nsew", pady=5)
        self.log_box = scrolledtext.ScrolledText(log_frame, width=42, height=25, state="disabled")
        self.log_box.pack(fill="both", expand=True)

    def _log_to_ui(self, message):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", message + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")
        self.trust_var.set(f"Trust Score: {self.logger.trust_score()}/100")

    def start(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self._log_to_ui("ERROR: Could not open camera.")
            return
        self.running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self._update_frame()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

    def _update_frame(self):
        if not self.running or self.cap is None:
            return
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame = self.engine.process_frame(frame)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb).resize((640, 480))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        self.root.after(20, self._update_frame)

    def on_close(self):
        self.stop()
        self.engine.close()
        self.logger.close()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = ProctoringApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()