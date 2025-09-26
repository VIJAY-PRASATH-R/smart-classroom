import cv2
import numpy as np
import time
import csv
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import mss
import threading
import tkinter as tk

# -----------------------------
# Load model and set labels
# -----------------------------
model = load_model('cnn_emotion_model.h5')
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
ENGAGEMENT_MAP = {
    "Happy": "Attentive",
    "Surprise": "Attentive",
    "Neutral": "Attentive",
    "Angry": "Frustrated",
    "Disgust": "Frustrated",
    "Fear": "Bored",
    "Sad": "Sleepy"
}

# -----------------------------
# Face detection
# -----------------------------
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# -----------------------------
# CSV logging
# -----------------------------
log_file = "engagement_log.csv"
with open(log_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Timestamp", "Attentive", "Bored", "Sleepy", "Frustrated"])

# -----------------------------
# Engagement counts (shared with GUI)
# -----------------------------
engagement_count = {"Attentive": 0, "Bored": 0, "Sleepy": 0, "Frustrated": 0}

# -----------------------------
# Tkinter pop-up setup
# -----------------------------
root = tk.Tk()
root.title("Engagement Monitor")
root.geometry("200x120+1000+50")  # small window in top-right
root.attributes("-topmost", True)  # always on top
root.resizable(False, False)

# --- Make window draggable ---
def make_draggable(widget):
    def on_drag_start(event):
        widget._drag_start_x = event.x
        widget._drag_start_y = event.y

    def on_drag_motion(event):
        x = widget.winfo_x() - widget._drag_start_x + event.x
        y = widget.winfo_y() - widget._drag_start_y + event.y
        widget.geometry(f"+{x}+{y}")

    widget.bind("<Button-1>", on_drag_start)
    widget.bind("<B1-Motion>", on_drag_motion)

make_draggable(root)

# --- Labels for engagement counts ---
labels = {}
y_pos = 10
for category in engagement_count.keys():
    lbl = tk.Label(root, text=f"{category}: 0", font=("Helvetica", 12))
    lbl.place(x=10, y=y_pos)
    labels[category] = lbl
    y_pos += 25

def update_gui():
    for category, lbl in labels.items():
        lbl.config(text=f"{category}: {engagement_count[category]}")
    root.after(500, update_gui)  # update every 0.5 sec

root.after(500, update_gui)

# -----------------------------
# Screen monitoring function
# -----------------------------
def monitor_screen():
    frame_count = 0
    with mss.mss() as sct:
        monitor_area = sct.monitors[1]  # capture primary monitor
        while True:
            screen = np.array(sct.grab(monitor_area))
            frame = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)

            if frame_count % 5 == 0:  # process every 5th frame
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_classifier.detectMultiScale(gray, 1.3, 5)

                # reset counts
                for key in engagement_count:
                    engagement_count[key] = 0

                for (x, y, w, h) in faces:
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_gray = cv2.resize(roi_gray, (48, 48))
                    roi = roi_gray.astype("float") / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)

                    preds = model.predict(roi, verbose=0)[0]
                    label = EMOTIONS[np.argmax(preds)]
                    engagement_label = ENGAGEMENT_MAP.get(label, "Attentive")
                    engagement_count[engagement_label] += 1

                # log to CSV
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                with open(log_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp,
                                     engagement_count["Attentive"],
                                     engagement_count["Bored"],
                                     engagement_count["Sleepy"],
                                     engagement_count["Frustrated"]])

            frame_count += 1
            time.sleep(0.2)  # small sleep to reduce CPU usage

# -----------------------------
# Start monitoring in thread
# -----------------------------
threading.Thread(target=monitor_screen, daemon=True).start()

# -----------------------------
# Start Tkinter loop
# -----------------------------
root.mainloop()
