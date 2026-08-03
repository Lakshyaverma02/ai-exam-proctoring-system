"""
Head Pose Estimation Program
=============================
Estimates a person's head orientation (pitch, yaw, roll) in real time
from a webcam feed, using MediaPipe Face Mesh for landmark detection
and OpenCV's solvePnP for 3D pose estimation.

Also classifies the gaze direction into a simple label: Center, Left,
Right, Up, or Down -- useful for attention/proctoring style monitoring.

Dependencies:
    pip install opencv-python mediapipe numpy

Usage:
    python head_pose.py
    python head_pose.py --camera 1

Press 'q' to quit.
"""

import cv2
import numpy as np
import argparse
import mediapipe as mp


# --------------------------------------------------------------------------- #
# 3D model points of key facial landmarks (generic head model, in mm),
# paired with their corresponding MediaPipe Face Mesh landmark indices.
# --------------------------------------------------------------------------- #
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),   # Right eye right corner
    (-150.0, -150.0, -125.0), # Left mouth corner
    (150.0, -150.0, -125.0),  # Right mouth corner
], dtype=np.float64)

LANDMARK_IDS = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_left_corner": 33,
    "right_eye_right_corner": 263,
    "left_mouth_corner": 61,
    "right_mouth_corner": 291,
}

YAW_THRESHOLD = 15.0     # degrees, left/right sensitivity
PITCH_THRESHOLD = 12.0   # degrees, up/down sensitivity


def get_image_points(landmarks, frame_w, frame_h):
    """Extract pixel coordinates of the landmarks used for pose estimation."""
    points = []
    for name in ["nose_tip", "chin", "left_eye_left_corner",
                 "right_eye_right_corner", "left_mouth_corner", "right_mouth_corner"]:
        idx = LANDMARK_IDS[name]
        lm = landmarks[idx]
        points.append((lm.x * frame_w, lm.y * frame_h))
    return np.array(points, dtype=np.float64)


def estimate_pose(image_points, frame_w, frame_h):
    """Run solvePnP and convert the rotation vector to Euler angles."""
    focal_length = frame_w
    center = (frame_w / 2, frame_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))  # assume no lens distortion

    success, rotation_vec, translation_vec = cv2.solvePnP(
        MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None, None, None, None, None

    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    pitch, yaw, roll = euler_angles.flatten()

    # normalize pitch into a human-readable range
    pitch = pitch - 180 if pitch > 90 else (pitch + 180 if pitch < -90 else pitch)

    return pitch, yaw, roll, rotation_vec, translation_vec


def classify_direction(pitch, yaw):
    if yaw > YAW_THRESHOLD:
        return "Looking Right"
    if yaw < -YAW_THRESHOLD:
        return "Looking Left"
    if pitch > PITCH_THRESHOLD:
        return "Looking Down"
    if pitch < -PITCH_THRESHOLD:
        return "Looking Up"
    return "Center"


def draw_axis(frame, image_points, rotation_vec, translation_vec, camera_matrix):
    """Draw a 3D axis on the nose tip to visualize orientation."""
    axis_len = 120.0
    axis_points_3d = np.array([
        [0, 0, 0],
        [axis_len, 0, 0],
        [0, -axis_len, 0],
        [0, 0, -axis_len],
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))
    projected, _ = cv2.projectPoints(
        axis_points_3d, rotation_vec, translation_vec, camera_matrix, dist_coeffs
    )
    projected = projected.reshape(-1, 2).astype(int)
    nose = tuple(projected[0])

    cv2.line(frame, nose, tuple(projected[1]), (0, 0, 255), 3)   # X - red
    cv2.line(frame, nose, tuple(projected[2]), (0, 255, 0), 3)   # Y - green
    cv2.line(frame, nose, tuple(projected[3]), (255, 0, 0), 3)   # Z - blue


def run(camera_index=0):
    mp_face_mesh = mp.solutions.face_mesh
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}.")

    print("Head pose estimation started. Press 'q' to quit.")
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera.")
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                image_points = get_image_points(landmarks, w, h)
                pitch, yaw, roll, rot_vec, trans_vec = estimate_pose(image_points, w, h)

                if pitch is not None:
                    direction = classify_direction(pitch, yaw)

                    focal_length = w
                    camera_matrix = np.array([
                        [focal_length, 0, w / 2],
                        [0, focal_length, h / 2],
                        [0, 0, 1],
                    ], dtype=np.float64)
                    draw_axis(frame, image_points, rot_vec, trans_vec, camera_matrix)

                    for (x, y) in image_points.astype(int):
                        cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

                    cv2.putText(frame, f"Pitch: {pitch:.1f}", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(frame, f"Yaw:   {yaw:.1f}", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(frame, f"Roll:  {roll:.1f}", (10, 75),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(frame, direction, (10, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No face detected", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Head Pose Estimation", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Head Pose Estimation Program")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    args = parser.parse_args()
    run(args.camera)


if __name__ == "__main__":
    main()