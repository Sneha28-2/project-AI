import cv2
import numpy as np
import pyautogui
import mediapipe as mp
from tkinter import *
import time

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Perform action based on detected gesture
def perform_action(gesture):
    if gesture == "left_swipe":
        pyautogui.hotkey('ctrl', 'left')  
    elif gesture == "right_swipe":
        pyautogui.hotkey('ctrl', 'right')  
    elif gesture == "zoom_in":
        pyautogui.hotkey('ctrl', '+')  
    elif gesture == "zoom_out":
        pyautogui.hotkey('ctrl', '-')  
    elif gesture == "click":
        pyautogui.click()  
    elif gesture == "thumb_up":
        print("Thumb Up Detected - Scroll Up")
        pyautogui.scroll(10)  
    elif gesture == "thumb_down":
        print("Thumb Down Detected - Scroll Down")
        pyautogui.scroll(-10)  
    elif gesture == "pointing_finger":
        print("Pointing Finger Detected - Moving Cursor")
        pyautogui.moveRel(20, 0)  
    elif gesture == "fist":
        print("Fist Detected - Minimize Window")
        pyautogui.hotkey('win', 'down')  
    elif gesture == "open_hand":
        print("Open Hand Detected - Maximize Window")
        pyautogui.hotkey('win', 'up')  

# Function to count fingers
def count_fingers(landmarks):
    # Get fingertip landmarks and base landmarks
    fingertips = [4, 8, 12, 16, 20]  # Thumb, index, middle, ring, pinky tips
    finger_bases = [2, 5, 9, 13, 17]  # For thumb, we'll use different logic
    
    # Count raised fingers
    count = 0
    
    # Special case for thumb
    # If thumb tip is to the left of thumb base for right hand (or right for left hand)
    if landmarks[fingertips[0]].x < landmarks[finger_bases[0]].x:
        count += 1
    
    # For the four fingers, we check if the tip is higher than the base joint
    for i in range(1, 5):
        if landmarks[fingertips[i]].y < landmarks[finger_bases[i]].y:
            count += 1
    
    return count

# Function to detect advanced gestures
def detect_advanced_gesture(landmarks):
    wrist = landmarks[0]
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    index_mcp = landmarks[5]  
    middle_mcp = landmarks[9]  

    gesture = "none"

    # Calculate distances
    thumb_index_dist = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_tip.x, index_tip.y]))
    index_middle_dist = np.linalg.norm(np.array([index_tip.x, index_tip.y]) - np.array([middle_tip.x, middle_tip.y]))

    # Thumb Up Gesture
    if thumb_tip.y < wrist.y and all(tip.y > wrist.y for tip in [index_tip, middle_tip, ring_tip, pinky_tip]):
        gesture = "thumb_up"

    # Thumb Down Gesture
    elif thumb_tip.y > wrist.y and all(tip.y > wrist.y for tip in [index_tip, middle_tip, ring_tip, pinky_tip]):
        gesture = "thumb_down"

    # OK Sign (Only Recognize, No Action)
    elif thumb_index_dist < 0.05 and all(tip.y < wrist.y for tip in [middle_tip, ring_tip, pinky_tip]):
        gesture = "okay"

    # Peace Sign (Only Recognize, No Action)
    elif index_tip.y < wrist.y and middle_tip.y < wrist.y and ring_tip.y > wrist.y and pinky_tip.y > wrist.y:
        gesture = "peace_sign"

    # Pointing Finger
    elif index_tip.y < wrist.y and all(tip.y > wrist.y for tip in [middle_tip, ring_tip, pinky_tip, thumb_tip]):
        gesture = "pointing_finger"

    # Crossed Fingers (Only Recognize, No Action)
    elif index_tip.y < wrist.y and middle_tip.y < wrist.y and index_middle_dist < 0.05:
        gesture = "crossed_fingers"

    return gesture

# Real-time webcam gesture prediction
def webcamPredict():
    cap = cv2.VideoCapture(0)  
    
    last_predicted_gesture = None
    last_process_time = time.time()
    processing_interval = 1.0  # Process every 3 seconds
    
    # Variables to store the last detected values
    last_finger_count = 0
    last_gesture = "none"
    time_until_next = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        img = cv2.flip(frame, 1)
        current_time = time.time()
        time_until_next = max(0, processing_interval - (current_time - last_process_time))
        
        # Process hand detection only every 3 seconds
        if current_time - last_process_time >= processing_interval:
            rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)
            
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Count fingers
                    last_finger_count = count_fingers(hand_landmarks.landmark)
                    
                    # Detect gestures
                    last_gesture = detect_advanced_gesture(hand_landmarks.landmark)
                    
                    # Only show Peace Sign, Crossed Fingers, and OK Sign (No Action)
                    if last_gesture in ["peace_sign", "crossed_fingers", "okay"]:
                        print(f"{last_gesture.replace('_', ' ').title()} Detected")  
                    elif last_gesture != "none" and last_gesture != last_predicted_gesture:
                        last_predicted_gesture = last_gesture
                        perform_action(last_gesture)
            else:
                # Reset when no hand detected
                last_finger_count = 0
                last_gesture = "No hand detected"
                
            last_process_time = current_time
        
        # Always display the last results (continuous display, even between processing)
        if last_gesture == "No hand detected":
            cv2.putText(img, "No hand detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(img, f"Gesture: {last_gesture}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, f"Fingers: {last_finger_count}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display countdown timer until next processing
        cv2.putText(img, f"Next scan in: {time_until_next:.1f}s", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        cv2.imshow("Hand Gesture Recognition", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# GUI Components
main = Tk()
main.title("Hand Gesture Recognition")
main.geometry("600x400")

font1 = ('times', 14, 'bold')
predictButton = Button(main, text="Recognize Gesture from Webcam", command=webcamPredict)
predictButton.place(x=50, y=100)
predictButton.config(font=font1)

main.mainloop()
#end of the program