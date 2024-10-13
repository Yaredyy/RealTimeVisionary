import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained model
# TODO change filename/filepath
model = load_model('best_model.keras')

# Preprocess the captured image to match the model's input format
def preprocess_image(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Resize the image to 28x28 pixels (as required by the model)
    resized = cv2.resize(gray, (28, 28))
    # Normalize the pixel values (0-255 to 0-1)
    normalized = resized.astype('float32') / 255.0
    # Reshape to match model's expected input (28x28x1)
    reshaped = np.expand_dims(normalized, axis=-1)
    return reshaped

# Function to find contours (potential digits) and predict
def find_and_recognize_digits(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to isolate the digits (simple binary threshold)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Get bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Ignore very small or large contours (filter out noise)
        if 50 < w < 300 and 50 < h < 300:
            # Extract the Region of Interest (ROI)
            roi = frame[y:y+h, x:x+w]

            # Preprocess the ROI for digit recognition
            preprocessed_image = preprocess_image(roi)

            # Expand dimensions to match the model's expected input
            input_data = np.expand_dims(preprocessed_image, axis=0)

            # Predict the digit
            predictions = model.predict(input_data, verbose=0)
            predicted_digit = np.argmax(predictions)

            # Draw a bounding box around the ROI and label it with the predicted digit
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{predicted_digit}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    return frame

# Define the function to capture and recognize digits from the entire screen
def recognize_digits():
    # Initialize the webcam (0 is usually the default camera)
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Process the frame to find and recognize digits
        processed_frame = find_and_recognize_digits(frame)

        # Display the frame with predictions
        cv2.imshow('Handwritten Digit Recognition', processed_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_digits()
