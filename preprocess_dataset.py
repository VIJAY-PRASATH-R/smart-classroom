import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical

# Path to FER-2013 CSV
csv_file = 'data/fer2013.csv'  # make sure this path is correct

# Load CSV
data = pd.read_csv(csv_file)

faces = []
labels = []

# Map emotions to your 4 classes
# Example: 0 = Bored, 1 = Sleepy, 2 = Frustrated, 3 = Doubt
# You may need to manually map FER-2013 emotions to your classes
emotion_map = {
    0: 0,  # Angry -> Frustrated
    1: 3,  # Disgust -> Doubt
    2: 3,  # Fear -> Doubt
    3: 1,  # Happy -> Sleepy
    4: 0,  # Sad -> Bored
    5: 3,  # Surprise -> Doubt
    6: 1   # Neutral -> Sleepy
}

for index, row in data.iterrows():
    pixels = row['pixels'].split()
    face = np.asarray(pixels, dtype='float32').reshape(48,48)
    face = face / 255.0
    faces.append(face)
    label = emotion_map[row['emotion']]
    labels.append(label)

faces = np.expand_dims(np.array(faces), -1)
labels = to_categorical(np.array(labels), num_classes=4)

# Save as NumPy arrays for training
np.save('faces.npy', faces)
np.save('labels.npy', labels)

print("Preprocessing done. Saved faces.npy and labels.npy")
