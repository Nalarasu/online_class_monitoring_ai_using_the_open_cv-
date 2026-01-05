import cv2
import numpy as np
import math

try:
    import mediapipe as mp
    # Prefer the classic solutions API when available
    try:
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
    except Exception:
        # Newer mediapipe installs expose the `tasks` API instead of `solutions`.
        # Raise a clear error with instructions to install a compatible mediapipe.
        raise ImportError
except ImportError:
    # Fall back to an OpenCV Haar-cascade based approach so the app can still run
    face_mesh = None
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Eye landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(landmarks, eye):
    A = np.linalg.norm(landmarks[eye[1]] - landmarks[eye[5]])
    B = np.linalg.norm(landmarks[eye[2]] - landmarks[eye[4]])
    C = np.linalg.norm(landmarks[eye[0]] - landmarks[eye[3]])
    return (A + B) / (2.0 * C)

def detect_attention(frame, closed_counter):
    status = "No Face"

    # If mediapipe face_mesh is available, use the original landmark-based logic
    if face_mesh is not None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:
            h, w, _ = frame.shape
            face = result.multi_face_landmarks[0]

            landmarks = np.array([
                [int(l.x * w), int(l.y * h)] for l in face.landmark
            ])

            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
            ear = (left_ear + right_ear) / 2

            # Sleeping logic
            if ear < 0.25:
                closed_counter += 1
            else:
                closed_counter = 0

            if closed_counter > 30:
                status = "Sleeping 😴"
            else:
                # Head pose (looking away)
                nose = landmarks[1]
                left = landmarks[234]
                right = landmarks[454]

                angle = abs(left[0] - right[0])
                if angle < 80:
                    status = "Not Listening 👀"
                else:
                    status = "Attentive 🙂"
    else:
        # Haar-cascade fallback: detect face and eyes, approximate EAR using box ratio
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) == 0:
            return status, closed_counter

        status = "Attentive 🙂"
        # look for eyes inside the first face
        (x, y, w, h) = faces[0]
        roi_gray = gray[y : y + h, x : x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        closed = 0
        for (ex, ey, ew, eh) in eyes:
            # approximate eye openness: height/width
            ratio = eh / float(ew)
            if ratio < 0.2:
                closed += 1

        if len(eyes) > 0:
            if closed >= len(eyes):
                closed_counter += 1
            else:
                closed_counter = 0

        if closed_counter > 30:
            status = "Sleeping 😴"

    return status, closed_counter
