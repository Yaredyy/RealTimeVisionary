# RealTimeVisionary

This is my collection of machine learning projects written in Python, focusing on real-time video processing and emotion detection. These projects utilize computer vision techniques to analyze video input from a camera.

## Projects

- **Emotion.py**

  - **Description**: Captures video from your webcam to detect faces and analyze facial expressions, identifying dominant emotions such as happy, sad, angry, fear, surprise, neutral, and disgust.
  - **Commands**:
    ```bash
    python src/Emotion.py
    ```
- **Peace.py**

  - **Description**: Detects hand gestures using the webcam and identifies specific symbols based on the positions of fingers (e.g., peace sign).
  - **Commands**:
    ```bash
    python src/Peace.py
    ```
- **DetectHumanInput.py**

  - **Description**: Combines face and hand detection to show both face and hand gestures in real-time on the video feed.
  - **Commands**:
    ```bash
    python src/DetectHumanInput.py
    ```
- **BasicInput.py**

  - **Description**: Displays a basic screen output using video input from the webcam without any advanced processing.
  - **Commands**:
    ```bash
    python src/BasicInput.py
    ```

## Requirements

To run these projects, you'll need the following Python packages:

- `opencv-python`
- `fer` (Facial Emotion Recognition)
- `mediapipe` (for hand gesture detection)

You can install the required packages using pip:

```bash
pip install -r requirements.txt
```
