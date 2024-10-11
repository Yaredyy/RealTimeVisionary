import cv2
from fer import FER

detector = FER()

def dominateEmotion(ar):
    max=-1
    emo= "no emotion"

    for emotion in ar:
       if (max<ar[emotion]):
        emo = emotion
        max = ar[emotion]
    
    return emo



# Capture video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect emotions
    emotions = detector.detect_emotions(frame)

    if emotions:
   # Add this line to see the detected emotions
        for emotion in emotions:

            # Draw the bounding box
            cv2.rectangle(frame, (emotion['box'][0], emotion['box'][1]),
                          (emotion['box'][0] + emotion['box'][2], emotion['box'][1] + emotion['box'][3]), (255, 0, 0), 2)


            cv2.putText(frame, dominateEmotion(emotion['emotions']), 
                        (emotion['box'][0], emotion['box'][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            

    else:
        cv2.putText(frame, "No faces detected", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


    cv2.imshow('Emotion Detection', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()