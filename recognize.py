import cv2
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "trainer.yml")

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        "trainer.yml not found. Train the model first."
    )

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    raise RuntimeError("Could not load Haar Cascade.")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

# LBPH distance threshold; tune this with testing
THRESHOLD = 45

print("[INFO] Face recognition started. Press 'q' to quit.")

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("[ERROR] Could not read webcam frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(50, 50)
        )

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (200, 200))

            person_id, confidence = recognizer.predict(face_img)

            if person_id == 1 and confidence < THRESHOLD:
                label = f"Authorized Owner ({confidence:.1f})"
                color = (0, 255, 0)
            else:
                label = f"Unknown ({confidence:.1f})"
                color = (0, 0, 255)

            cv2.rectangle(
                frame, (x, y), (x+w, y+h), color, 2
            )
            cv2.putText(
                frame, label, (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()