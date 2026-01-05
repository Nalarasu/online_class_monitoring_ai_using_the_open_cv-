import cv2
from attention import detect_attention

cap = cv2.VideoCapture(0)
closed_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    status, closed_counter = detect_attention(frame, closed_counter)

    cv2.putText(frame, f"Status: {status}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    cv2.imshow("Online Class Monitoring AI", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
