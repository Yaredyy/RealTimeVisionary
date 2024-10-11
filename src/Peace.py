import cv2
import mediapipe as mp


# Method for detecting gestures
def detect_gesture(hand_landmarks, frame_height):
    # Get the landmark positions
    landmarks = hand_landmarks.landmark
    index_finger_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_finger_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    pinki_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]

    # Convert normalized coordinates to pixel values
    index_y = int(index_finger_tip.y * frame_height)
    thumb_y = int(thumb_tip.y * frame_height)
    middle_y = int(middle_finger_tip.y*frame_height)
    pink_y = int(pinki_tip.y*frame_height)

    # Define gesture conditions
    if (
        abs(index_y - thumb_y) < 100
    )and (abs(middle_y-index_y)>100):
        if(abs(pink_y-middle_y)<100):
            return "Peace Sign"
        else:
            return "dont do that"
    
    
    
    
    return "Unknown Gesture"


# Initialize face detection or hand tracking models
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Setup video capture
cap = cv2.VideoCapture(0)

# Initialize models
with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5
) as face_detection, mp_hands.Hands(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        frame_height, frame_width, _ = frame.shape
        if not ret:
            break

        # Convert frame to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Face detection
        results_face = face_detection.process(image)

        # Hand tracking
        results_hands = hands.process(image)
        results_hands
        # Draw face detection landmarks
        if results_face.detections:
            for detection in results_face.detections:
                mp_drawing.draw_detection(frame, detection)

        # Draw hand landmarks
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
                gesture = detect_gesture(hand_landmarks, frame_height)
                cv2.putText(
                    frame,
                    gesture,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

        # Display the output
        cv2.imshow("Face and Hand Detection", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
