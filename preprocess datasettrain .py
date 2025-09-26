import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# Paths
data_dir = r"C:\Users\vijay\Desktop\dnn project\smart classroom\data\processed"
emotions = ["Bored", "Sleepy", "Frustrated", "Doubt"]

# Load images and labels
images = []
labels = []

for idx, emotion in enumerate(emotions):
    emotion_path = os.path.join(data_dir, emotion)
    for img_file in os.listdir(emotion_path):
        img_path = os.path.join(emotion_path, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (48,48))
        img = img.astype('float32')/255.0
        images.append(img)
        labels.append(idx)

images = np.array(images)
images = np.expand_dims(images, -1)  # add channel dimension
labels = to_categorical(labels, num_classes=len(emotions))

# Create CNN model
model = Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(256, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(len(emotions), activation='softmax')
])

model.compile(optimizer=Adam(0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Save best model during training
checkpoint = ModelCheckpoint(r"C:\Users\vijay\Desktop\dnn project\smart classroom\train\emotion_model.h5", monitor='val_accuracy', save_best_only=True)

# Train model
model.fit(images, labels, epochs=30, batch_size=64, validation_split=0.1, callbacks=[checkpoint])

print("Training complete! Model saved as emotion_model.h5")

