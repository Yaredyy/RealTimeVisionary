import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('best_handwritten_digit_model.keras')

def preprocess_image(img):
    """Preprocess the image for digit recognition."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if h > w:
        new_h, new_w = 28, int(28 * (w / h))
    else:
        new_h, new_w = int(28 * (h / w)), 28
    resized = cv2.resize(gray, (new_w, new_h))

    padded = np.zeros((28, 28), dtype=np.float32)
    y_offset = (28 - new_h) // 2
    x_offset = (28 - new_w) // 2
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized / 255.0
    return np.expand_dims(padded, axis=-1)

def find_and_recognize_digits(frame):
    """Detect and recognize digits in the frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 20 < w < 300 and 20 < h < 300:
            roi = frame[y:y+h, x:x+w]
            preprocessed_image = preprocess_image(roi)
            input_data = np.expand_dims(preprocessed_image, axis=0)
            predictions = model.predict(input_data, verbose=0)
            predicted_digit = np.argmax(predictions)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{predicted_digit}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    return frame

def main():
    """Capture video and recognize digits."""
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame = find_and_recognize_digits(frame)
        cv2.imshow("Digit Recognition", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
