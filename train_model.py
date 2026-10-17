import cv2
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "user_1")
MODEL_PATH = os.path.join(BASE_DIR, "trainer.yml")

def main():
    if not hasattr(cv2, "face"):
        raise RuntimeError(
            "cv2.face is missing. Install opencv-contrib-python "
            "in the active Python environment."
        )

    if not os.path.isdir(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_PATH}"
        )

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    face_samples = []
    ids = []

    for filename in os.listdir(DATASET_PATH):
        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        ):
            continue

        image_path = os.path.join(DATASET_PATH, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"[WARNING] Could not read: {filename}")
            continue

        # The collection script already cropped the face.
        image = cv2.resize(image, (200, 200))

        face_samples.append(image)
        ids.append(1)  # Owner ID

    if len(face_samples) < 2:
        raise RuntimeError(
            "Need at least 2 readable face images. "
            "Run collect_data.py first."
        )

    print(f"[INFO] Training with {len(face_samples)} images...")

    recognizer.train(
        face_samples,
        np.array(ids, dtype=np.int32)
    )

    recognizer.save(MODEL_PATH)

    print("[INFO] Training completed successfully!")
    print(f"[INFO] Images used: {len(face_samples)}")
    print(f"[INFO] Model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    main()