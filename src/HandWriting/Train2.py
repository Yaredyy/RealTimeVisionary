import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from PIL import Image
from tqdm import tqdm

# Load dataset function (same as yours)
def load_data(data_dir):
    X = []
    y = []
    
    print("Loading dataset...")
    for label in range(10):  # Assuming digits 0-9
        label_dir = os.path.join(data_dir, str(label))
        label_dir = os.path.join(label_dir, str(label))
        
        print(f"Processing images for label: {label}")
        for filename in tqdm(os.listdir(label_dir)):  # Show progress
            img_path = os.path.join(label_dir, filename)
            img = Image.open(img_path).convert('L')
            img = img.resize((28, 28))
            
            img = np.array(img, dtype='float32') / 255.0  # Normalize the image
            
            X.append(img)
            y.append(label)

    print("Dataset loaded successfully.")
    return np.array(X), np.array(y)

# Define paths
data_dir = '/Users/yaredyohannes/Documents/GitHub/RealTimeVisionary/src/HandWriting/dataset'  # Update this to your dataset path
X, y = load_data(data_dir)

# Reshape data
X = np.array(X)
X = np.expand_dims(X, axis=-1)  # Add channel dimension for grayscale images
y_categorical = to_categorical(y, num_classes=10)

# Model definition (matching theirs)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compile model (same as theirs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Learning rate scheduling
def lr_schedule(epoch):
    initial_lr = 0.0001
    decay = 0.95
    return initial_lr * (decay ** epoch)

lr_scheduler = LearningRateScheduler(lr_schedule)

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train model (matching their batch size and epochs)
history = model.fit(
    X, 
    y_categorical,
    batch_size=32,  # Smaller batch size, like theirs
    epochs=20,
    validation_split=0.2,  # 20% of the data used for validation
    callbacks=[early_stopping, lr_scheduler]
)

# Save final model
model.save('final_handwritten_digit_model.keras')

# Plotting accuracy (same as theirs)
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.legend()
plt.show()

# Plotting loss (same as theirs)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.legend()
plt.show()
