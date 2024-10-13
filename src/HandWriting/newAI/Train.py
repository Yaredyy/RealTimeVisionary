import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Step 1: Load the Dataset
def load_dataset(data_path):
    images = []
    labels = []

    # Loop through each digit folder
    for digit in range(10):
        digit_folder = os.path.join(data_path, str(digit))
        digit_folder = os.path.join(digit_folder, str(digit))
        if not os.path.exists(digit_folder):
            print(f"Folder for digit {digit} does not exist.")
            continue
        else:
            print("exists")

        for filename in os.listdir(digit_folder):
            if filename.endswith('.png'):
                img_path = os.path.join(digit_folder, filename)
                img = load_img(img_path, color_mode='grayscale', target_size=(28, 28))
                img = img_to_array(img) / 255.0  # Normalize the image
                images.append(img)
                labels.append(digit)

    print(f"Loaded {len(images)} images.")
    return np.array(images), np.array(labels)


# Example usage of load_dataset
data_path = '/Users/yaredyohannes/Documents/GitHub/RealTimeVisionary/src/HandWriting/dataset'  # Change this to your dataset path
X, y = load_dataset(data_path)

# Step 2: Prepare Data for Training
y_categorical = to_categorical(y, num_classes=10)

# Step 3: Create and Train the Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y_categorical, epochs=10, batch_size=32, validation_split=0.2)

# Step 4: Save the Trained Model
model.save('digit_recognizer.h5')

print("Model training complete and saved as 'digit_recognizer.h5'")
