import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
ratio_to_inches = 2.0

def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue
    
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        finger_tips = []
        
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            
            index_tip_coords = (int(index_tip.x * image.shape[1]), int(index_tip.y * image.shape[0]))
            wrist_coords = (int(wrist.x * image.shape[1]), int(wrist.y * image.shape[0]))
            
            finger_tips.append(index_tip_coords)
            hand_size = calculate_distance(index_tip_coords, wrist_coords)
        
        if len(finger_tips) == 2 and hand_size > 0:
            pixel_distance = calculate_distance(finger_tips[0], finger_tips[1])
            distance_in_inches = pixel_distance / hand_size * 2.25 * ratio_to_inches
            cv2.line(image, finger_tips[0], finger_tips[1], (0, 255, 0), 2)
            mid_point = (int((finger_tips[0][0] + finger_tips[1][0]) / 2), int((finger_tips[0][1] + finger_tips[1][1]) / 2) - 10)
            cv2.putText(image, f"{distance_in_inches:.2f} inches", mid_point, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Hand Distance Measurement', image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
