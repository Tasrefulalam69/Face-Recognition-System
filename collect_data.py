import cv2
import os
import sys

# Create dataset directory
dataset_path = os.path.join("dataset", "user_1")
os.makedirs(dataset_path, exist_ok=True)

# Check that the correct OpenCV package is loaded
if not hasattr(cv2, "CascadeClassifier") or not hasattr(cv2, "data"):
    raise RuntimeError(
        "OpenCV is not loaded correctly. "
        "Follow the installation steps below."
    )

# Load the Haar Cascade face detector
cascade_path = os.path.join(
    cv2.data.haarcascades,
    "haarcascade_frontalface_default.xml"
)

face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    raise RuntimeError(
        f"Could not load face detector: {cascade_path}"
    )

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError(
        "Could not access webcam. Check camera permissions "
        "or try changing VideoCapture(0) to VideoCapture(1)."
    )

count = 0
max_images = 30
frame_number = 0
capture_interval = 5

print("[INFO] Capturing face data...")
print("[INFO] Look at the camera. Press 'q' to quit.")

try:
    while count < max_images:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("[ERROR] Could not read webcam frame.")
            break

        frame_number += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(50, 50)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Capture one face every 5 frames
            if (
                frame_number % capture_interval == 0
                and count < max_images
            ):
                face_img = gray[y:y + h, x:x + w]

                file_path = os.path.join(
                    dataset_path,
                    f"{count + 1}.jpg"
                )

                if cv2.imwrite(file_path, face_img):
                    count += 1
                    print(f"[INFO] Saved image {count}/{max_images}")

        cv2.putText(
            frame,
            f"Captured: {count}/{max_images}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow("Capture Face Data", frame)

        # Check the quit key once per frame
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

print(
    f"[INFO] Saved {count} face images in '{dataset_path}'."
)