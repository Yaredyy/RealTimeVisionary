import cv2
import mediapipe as mp

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

        # Display the output
        cv2.imshow("Face and Hand Detection", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
