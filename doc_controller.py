import cv2
import mediapipe as mp
import pyautogui
import platform
import time

# --- CONFIGURATION ---
# Check OS to set correct shortcuts
CURRENT_OS = platform.system()
print(f"Detected OS: {CURRENT_OS}")

# Sensitivity settings
SCROLL_SENSITIVITY = 5  # Higher = faster scroll
SCROLL_THRESHOLD = 0.02 # Minimum movement to trigger scroll (noise filter)
GESTURE_COOLDOWN = 1.0  # Seconds to wait between minimize triggers

# MediaPipe Setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Global Variables
prev_y = 0  # To track previous Y position for scrolling
previous_fingers_up = 0 # To track history for "Palm -> Fist"
last_action_time = 0

def get_fingers_status(landmarks):
    """
    Returns a list of 5 booleans [Thumb, Index, Middle, Ring, Pinky]
    True = Extended, False = Curled
    """
    fingers = []
    
    # Thumb: Check if tip is to the right/left of the knuckle (depending on hand side)
    # Simplified: Check x-distance logic or just ignore thumb for simple scroll
    # Here we use a simplified logic: if tip.x is 'outside' the knuckle. 
    # For robustness in this specific demo, we often rely mostly on the 4 fingers.
    # Let's use the standard y-axis check for fingers 2-5.
    
    # Tips Ids: [4, 8, 12, 16, 20]
    # PIP Ids (Knuckles): [2, 6, 10, 14, 18]
    
    # Thumb (Tip 4 vs IP 3) - Logic varies by hand side, simplified here:
    if landmarks[4].x < landmarks[3].x: # Right hand thumb check (approx)
        fingers.append(True)
    else:
        fingers.append(False)

    # 4 Fingers (Index to Pinky)
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        if landmarks[tip].y < landmarks[pip].y: # Note: Y decreases going UP in CV2
            fingers.append(True)
        else:
            fingers.append(False)
            
    return fingers

def count_fingers_up(fingers_list):
    # We define "Open Palm" as 4 or 5 fingers up
    # We define "Fist" as 0 or 1 finger up (thumb sometimes stays out)
    count = sum(fingers_list) # simple sum of Trues
    return count

def perform_minimize():
    """Execute minimize shortcut based on OS"""
    print("Action: Minimize Window")
    if CURRENT_OS == "Darwin": # macOS
        pyautogui.hotkey('command', 'm') 
    elif CURRENT_OS == "Windows":
        pyautogui.hotkey('win', 'd') # Toggles Desktop
    else:
        print("OS not supported for automation shortcuts")

# --- MAIN LOOP ---
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 1. Preprocessing
    frame = cv2.flip(frame, 1) # Mirror image for natural feel
    h, w, c = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 2. Process Hand
    results = hands.process(rgb_frame)
    
    status_text = "Status: Idle"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get landmarks list
            lm_list = hand_landmarks.landmark
            
            # Analyze Gesture
            fingers_status = get_fingers_status(lm_list)
            total_up = count_fingers_up(fingers_status[1:]) # Count index-pinky (ignore thumb for robustness)
            
            # --- LOGIC 1: MINIMIZE (Open Palm -> Fist) ---
            # We look for a state transition. 
            # If previously we had >3 fingers up, and now we have 0.
            current_time = time.time()
            if total_up == 0 and previous_fingers_up >= 4:
                if current_time - last_action_time > GESTURE_COOLDOWN:
                    perform_minimize()
                    last_action_time = current_time
                    status_text = "ACTION: Minimize!"
            
            previous_fingers_up = total_up

            # --- LOGIC 2: SCROLL (Index + Middle Up Only) ---
            # Pattern: Index(True), Middle(True), Ring(False), Pinky(False)
            if fingers_status[1] and fingers_status[2] and not fingers_status[3] and not fingers_status[4]:
                status_text = "Mode: Scrolling"
                
                # Use Tip of Index Finger (Landmark 8) for tracking
                current_y = lm_list[8].y 
                
                # Check delta
                if prev_y != 0: # Skip first frame
                    diff = current_y - prev_y
                    
                    if abs(diff) > SCROLL_THRESHOLD:
                        # Map movement to scroll units
                        # Multiplier depends on OS sensitivity
                        scroll_amount = int(diff * 100 * SCROLL_SENSITIVITY)
                        
                        if CURRENT_OS == "Darwin":
                            scroll_amount *= 2 # Mac often needs different scaling
                        else:
                            scroll_amount *= 5 # Windows often needs larger integers
                            
                        # INVERT: Moving hand UP (negative Y diff) should scroll UP (positive val)
                        # PyAutoGUI scroll: +ve is UP, -ve is DOWN
                        # Our `diff` is (New - Old). 
                        # If Hand moves DOWN, Y increases, diff is +ve. We want Scroll DOWN (-ve).
                        
                        pyautogui.scroll(-scroll_amount)
                        
                prev_y = current_y
            else:
                prev_y = 0 # Reset tracking if gesture breaks

    # UI Feedback
    cv2.putText(frame, status_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Hand Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == 27: # Esc to quit
        break

cap.release()
cv2.destroyAllWindows()