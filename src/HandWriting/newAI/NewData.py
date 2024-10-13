# Imports

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input,Conv2D, Dense, Flatten ,Dropout ,MaxPooling2D
from tensorflow.keras.models import Model,Sequential
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping,LearningRateScheduler

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
sample_submission=pd.read_csv('sample_submission.csv')

# Visualizing the labels

plt.figure(figsize=(10,5))
sns.countplot(data=train,x='label',palette='CMRmap')

# Reshaping the data

X=train.drop('label',axis=1)
y=train['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)

X_train.shape[0]

#Reshaping the data to be suitable for the neural network.

X_train=X_train.values.reshape(X_train.shape[0],28,28,1)
X_test = X_test.values.reshape(X_test.shape[0], 28, 28, 1)
test = test.values.reshape(test.shape[0], 28, 28, 1)

print(X_train.shape)
print(X_test.shape)
print(test.shape)

## Normalization

input_shape=(28,28,1)
X_train=X_train.astype("float32")
X_test=X_test.astype("float32")
test=test.astype("float32")

#normalizing data to range from (0 to 255) to (0 to 1)

X_train/=255
X_test/=255
test/=255

print('x_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


# Encoding

#One hot encoding our labels to be represented as columns of 0s and 1s.

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print("Number of classes: " + str(y_test.shape[1]))


#Classyfing our data in terms of classes and number of pixels

num_classes=y_test.shape[1]
num_pixels=X_train.shape[1]+X_train.shape[2]

# Model 

#Feel free to expirement with the modelling by adding more layers or using advanced techniques!

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


X_train.shape

y_train.shape

#Applying early stopping and learning rate scheduling to deal with overfitting and improve results.

early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)

# Define learning rate schedule function
def lr_schedule(epoch):
    initial_lr = 0.0001  # Initial learning rate
    decay = 0.95  
    lr = initial_lr * (decay ** epoch)
    return lr

lr_scheduler = LearningRateScheduler(lr_schedule)

history = model.fit(X_train,
                    y_train,
                    batch_size=32,
                    epochs=20,
                    verbose=1,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping, lr_scheduler])

score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])


plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#Validation's loss and accuracy converges to the training's loss and accuracy, which means the model is performing very well!

# Submission

predictions = model.predict(test)
predictions = np.argmax(predictions,axis = 1)
predictions = pd.Series(predictions,name = "Label")

sample_submission['Label'] = predictions
sample_submission.head()

sample_submission.to_csv("submission.csv", index = False)