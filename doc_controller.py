import cv2
import time
import platform
import pyautogui
import mediapipe as mp

from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

# ================= SYSTEM =================
OS_NAME = platform.system()
MINIMIZE_COOLDOWN = 1.2
SCROLL_SENSITIVITY = 300
SCROLL_THRESHOLD = 0.015

# ================= MEDIAPIPE TASKS =================
base_options = BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)

landmarker = vision.HandLandmarker.create_from_options(options)

# ================= STATE =================
prev_finger_count = 0
last_minimize_time = 0
prev_scroll_y = None

# ================= HELPERS =================
def count_fingers(landmarks):
    fingers = []

    # Thumb ignored (mac camera angle issues)
    fingers.append(False)

    finger_pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]
    for tip, pip in finger_pairs:
        fingers.append(landmarks[tip].y < landmarks[pip].y)

    return fingers

def minimize_window():
    if OS_NAME == "Darwin":
        pyautogui.hotkey("command", "m")
    elif OS_NAME == "Windows":
        pyautogui.hotkey("win", "d")

# ================= CAMERA =================
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    status = "Idle"

    if result.hand_landmarks:
        lm = result.hand_landmarks[0]

        fingers = count_fingers(lm)
        finger_count = sum(fingers)
        now = time.time()

        # ===== GESTURE 1: MINIMIZE =====
        if finger_count == 0 and prev_finger_count >= 4:
            if now - last_minimize_time > MINIMIZE_COOLDOWN:
                minimize_window()
                last_minimize_time = now
                status = "ACTION: Minimize"

        prev_finger_count = finger_count

        # ===== GESTURE 2: SCROLL =====
        if fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
            status = "Mode: Scroll"
            current_y = lm[8].y

            if prev_scroll_y is not None:
                delta = current_y - prev_scroll_y
                if abs(delta) > SCROLL_THRESHOLD:
                    scroll_val = int(-delta * SCROLL_SENSITIVITY)
                    pyautogui.scroll(scroll_val)

            prev_scroll_y = current_y
        else:
            prev_scroll_y = None

    cv2.putText(frame, status, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    cv2.imshow("Gesture Controller", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
