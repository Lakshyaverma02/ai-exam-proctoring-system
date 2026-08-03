"""
Exam Proctoring Alert System
=============================
Monitors a webcam feed during an online exam and raises alerts for
suspicious behaviour:

  1. No face detected        -> student may have left the frame
  2. Multiple faces detected -> possible unauthorized assistance
  3. Face looking away       -> possible cheating (reading notes, etc.)
  4. Face too close / too far-> possible camera tampering
  5. Prolonged silence check hooks (optional, for audio module)

All alerts are timestamped, printed to the console, written to a log
file (proctoring_log.csv), and an in-memory alert counter is kept so a
"trust score" can be reported at the end of the session.

Dependencies:
    pip install opencv-python numpy

Usage:
    python proctoring_alert_system.py --camera 0

Press 'q' to stop monitoring.
"""

import cv2
import time
import csv
import os
import argparse
from datetime import datetime
from collections import deque


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
class Config:
    NO_FACE_ALERT_SECONDS = 3          # how long a face must be missing before alerting
    LOOK_AWAY_ALERT_SECONDS = 3        # how long "looking away" must persist
    MULTI_FACE_COOLDOWN = 5            # seconds between repeated multi-face alerts
    NO_FACE_COOLDOWN = 5
    LOOK_AWAY_COOLDOWN = 5
    FACE_CENTER_TOLERANCE = 0.18       # fraction of frame width considered "centered"
    MIN_FACE_AREA_RATIO = 0.02         # face too small -> too far from camera
    MAX_FACE_AREA_RATIO = 0.55         # face too large -> too close / covering camera
    LOG_FILE = "proctoring_log.csv"


# --------------------------------------------------------------------------- #
# Alert Logger
# --------------------------------------------------------------------------- #
class AlertLogger:
    def __init__(self, log_file):
        self.log_file = log_file
        self.counts = {}
        is_new = not os.path.exists(log_file)
        self._file = open(log_file, "a", newline="")
        self._writer = csv.writer(self._file)
        if is_new:
            self._writer.writerow(["timestamp", "alert_type", "message"])

    def raise_alert(self, alert_type, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[ALERT] {timestamp} | {alert_type} | {message}")
        self._writer.writerow([timestamp, alert_type, message])
        self._file.flush()
        self.counts[alert_type] = self.counts.get(alert_type, 0) + 1

    def summary(self):
        print("\n--- Session Summary ---")
        if not self.counts:
            print("No alerts raised. Clean session.")
        for alert_type, count in self.counts.items():
            print(f"{alert_type}: {count}")
        total = sum(self.counts.values())
        # simple trust score: starts at 100, loses points per alert type weighted
        weights = {
            "NO_FACE": 5,
            "MULTIPLE_FACES": 10,
            "LOOKING_AWAY": 3,
            "CAMERA_TAMPER": 8,
        }
        penalty = sum(self.counts.get(k, 0) * w for k, w in weights.items())
        trust_score = max(0, 100 - penalty)
        print(f"Total alerts: {total}")
        print(f"Trust score: {trust_score}/100")

    def close(self):
        self._file.close()


# --------------------------------------------------------------------------- #
# Proctoring Monitor
# --------------------------------------------------------------------------- #
class ProctoringMonitor:
    def __init__(self, camera_index=0, show_video=True):
        self.cfg = Config()
        self.logger = AlertLogger(self.cfg.LOG_FILE)
        self.show_video = show_video

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError("Could not load Haar cascade for face detection.")

        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {camera_index}.")

        # State tracking
        self.no_face_since = None
        self.look_away_since = None
        self.last_multi_face_alert = 0
        self.last_no_face_alert = 0
        self.last_look_away_alert = 0
        self.last_tamper_alert = 0

        # smoothing buffer for face count, to avoid flicker false positives
        self.face_count_buffer = deque(maxlen=5)

    def _detect_faces(self, gray_frame):
        faces = self.face_cascade.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60)
        )
        return faces

    def _check_face_count(self, faces):
        now = time.time()
        self.face_count_buffer.append(len(faces))
        stable_count = max(set(self.face_count_buffer), key=self.face_count_buffer.count)

        if stable_count == 0:
            if self.no_face_since is None:
                self.no_face_since = now
            elapsed = now - self.no_face_since
            if elapsed >= self.cfg.NO_FACE_ALERT_SECONDS and \
               now - self.last_no_face_alert > self.cfg.NO_FACE_COOLDOWN:
                self.logger.raise_alert(
                    "NO_FACE", f"No face detected for {elapsed:.1f}s. Student may have left."
                )
                self.last_no_face_alert = now
        else:
            self.no_face_since = None

        if stable_count > 1 and now - self.last_multi_face_alert > self.cfg.MULTI_FACE_COOLDOWN:
            self.logger.raise_alert(
                "MULTIPLE_FACES", f"{stable_count} faces detected. Possible unauthorized assistance."
            )
            self.last_multi_face_alert = now

    def _check_gaze_and_distance(self, faces, frame_width, frame_height):
        now = time.time()
        if len(faces) != 1:
            self.look_away_since = None
            return

        x, y, w, h = faces[0]
        face_center_x = x + w / 2
        frame_center_x = frame_width / 2
        offset_ratio = abs(face_center_x - frame_center_x) / frame_width

        if offset_ratio > self.cfg.FACE_CENTER_TOLERANCE:
            if self.look_away_since is None:
                self.look_away_since = now
            elapsed = now - self.look_away_since
            if elapsed >= self.cfg.LOOK_AWAY_ALERT_SECONDS and \
               now - self.last_look_away_alert > self.cfg.LOOK_AWAY_COOLDOWN:
                self.logger.raise_alert(
                    "LOOKING_AWAY", f"Face off-center for {elapsed:.1f}s. Possible looking away."
                )
                self.last_look_away_alert = now
        else:
            self.look_away_since = None

        face_area_ratio = (w * h) / (frame_width * frame_height)
        if face_area_ratio < self.cfg.MIN_FACE_AREA_RATIO or \
           face_area_ratio > self.cfg.MAX_FACE_AREA_RATIO:
            if now - self.last_tamper_alert > self.cfg.NO_FACE_COOLDOWN:
                self.logger.raise_alert(
                    "CAMERA_TAMPER",
                    f"Unusual face size ratio ({face_area_ratio:.2f}). Check camera distance/obstruction."
                )
                self.last_tamper_alert = now

    def run(self):
        print("Proctoring session started. Press 'q' to stop.")
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Camera read failed. Stopping.")
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self._detect_faces(gray)
                h, w = frame.shape[:2]

                self._check_face_count(faces)
                self._check_gaze_and_distance(faces, w, h)

                if self.show_video:
                    for (fx, fy, fw, fh) in faces:
                        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
                    cv2.putText(frame, "Proctoring Active", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("Exam Proctoring", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.logger.summary()
        self.logger.close()
        print(f"Log saved to: {os.path.abspath(self.cfg.LOG_FILE)}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Exam Proctoring Alert System")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--no-video", action="store_true", help="Run without displaying video window")
    args = parser.parse_args()

    monitor = ProctoringMonitor(camera_index=args.camera, show_video=not args.no_video)
    monitor.run()


if __name__ == "__main__":
    main()