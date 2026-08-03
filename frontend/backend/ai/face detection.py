"""
Face Detection Program
=======================
Detects faces in real time from a webcam feed (default) or in a static
image file, using OpenCV's Haar Cascade classifier.

Dependencies:
    pip install opencv-python

Usage:
    Webcam mode (default):
        python face_detection.py

    Image mode:
        python face_detection.py --image path/to/photo.jpg

    Custom camera index:
        python face_detection.py --camera 1

Press 'q' to quit webcam mode. In image mode, press any key to close.
"""

import cv2
import argparse


def load_face_cascade():
    """Load OpenCV's built-in frontal-face Haar cascade."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError("Failed to load Haar cascade classifier.")
    return cascade


def detect_faces(cascade, gray_frame):
    """Return list of (x, y, w, h) bounding boxes for detected faces."""
    return cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(40, 40),
    )


def draw_faces(frame, faces):
    """Draw rectangles and a count label on the frame."""
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(
        frame,
        f"Faces detected: {len(faces)}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    return frame


def run_webcam(camera_index=0):
    cascade = load_face_cascade()
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}.")

    print("Webcam face detection started. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(cascade, gray)
        frame = draw_faces(frame, faces)

        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_image(image_path):
    cascade = load_face_cascade()
    frame = cv2.imread(image_path)
    if frame is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(cascade, gray)
    frame = draw_faces(frame, faces)

    print(f"{len(faces)} face(s) detected in {image_path}")

    output_path = "detected_" + image_path.split("/")[-1]
    cv2.imwrite(output_path, frame)
    print(f"Result saved to {output_path}")

    cv2.imshow("Face Detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Face Detection Program")
    parser.add_argument("--image", type=str, help="Path to an image file (omit for webcam mode)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index for webcam mode (default: 0)")
    args = parser.parse_args()

    if args.image:
        run_image(args.image)
    else:
        run_webcam(args.camera)


if __name__ == "__main__":
    main()