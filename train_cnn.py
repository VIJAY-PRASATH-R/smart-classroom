import numpy as np
from cnn_emotion_classifier import create_emotion_model
from tensorflow.keras.callbacks import ModelCheckpoint

# Load preprocessed data
faces = np.load('faces.npy')
labels = np.load('labels.npy')

# Create CNN model
model = create_emotion_model()

# Save best model during training
checkpoint = ModelCheckpoint("emotion_model.h5", monitor='val_accuracy', save_best_only=True)

# Train the model
model.fit(faces, labels, epochs=30, batch_size=64, validation_split=0.1, callbacks=[checkpoint])
