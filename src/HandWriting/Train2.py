import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm

# Load dataset function
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
            img = plt.imread(img_path)

            # Convert image to grayscale if it's RGBA or RGB
            if img.ndim == 3:  # Check if it's a color image
                img = np.mean(img[..., :3], axis=-1)  # Convert to grayscale
            img = img.astype('float32') / 255.0  # Normalize the image
            img = np.resize(img, (28, 28))  # Ensure correct size
            X.append(img)
            y.append(label)

    print("Dataset loaded successfully.")
    return np.array(X), np.array(y)

# Define paths
data_dir = '/Users/yaredyohannes/Documents/GitHub/RealTimeVisionary/src/HandWriting/dataset'  # Update this to your dataset path
X, y = load_data(data_dir)

# Normalize images
X = np.array(X)
X = np.expand_dims(X, axis=-1)  # Add channel dimension for grayscale images
y_categorical = to_categorical(y, num_classes=10)

# Refined data augmentation to avoid harmful transformations
datagen = ImageDataGenerator(
    validation_split=0.2  # Only use validation split, no augmentation
)

# Model definition
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compile model with reduced learning rate for stability
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks for early stopping and model checkpointing
checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min')
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Fit model with validation data and visual feedback
model.fit(datagen.flow(X, y_categorical, batch_size=32, subset='training'),
          epochs=20,
          validation_data=datagen.flow(X, y_categorical, batch_size=32, subset='validation'),
          callbacks=[checkpoint, early_stop])

# Save final model
model.save('final_handwritten_digit_model.keras')
