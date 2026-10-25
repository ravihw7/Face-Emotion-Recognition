import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model (compile=False suppresses the metrics warning)
model = load_model("models/emotion_model.h5", compile=False)

# Emotion labels — must match alphabetical folder order from training:
# {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}
emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

# Load face detector
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

# Start webcam
cap = cv2.VideoCapture(0)

print("Emotion Recognition running. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face / 255.0
        face = np.reshape(face, (1, 48, 48, 1))

        # Use predict with verbose=0 to suppress progress bars
        prediction = model.predict(face, verbose=0)
        emotion_idx = np.argmax(prediction)
        confidence = prediction[0][emotion_idx] * 100
        emotion = emotion_labels[emotion_idx]

        # Draw rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (255, 0, 0),
            2
        )

        # Put emotion label with confidence
        label = f"{emotion} ({confidence:.1f}%)"
        cv2.putText(
            frame,
            label,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    cv2.imshow("Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()