from keras.models import load_model
from cryptography.fernet import Fernet
from keras.preprocessing.image import img_to_array
from datetime import datetime, timedelta
from pymongo import MongoClient
import hashlib
import pandas as pd
import cv2
import numpy as np

# --- Load Face Detection and Emotion Model ---
face_classifier = cv2.CascadeClassifier(r'C:\Users\91994\project\projectds\haarcascade_frontalface_default.xml')
classifier = load_model(r'C:\Users\91994\project\projectds\Emotion_little_vgg.h5')

# --- Emotion Labels and Task Mapping ---
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
TASK_MAP = {
    "Happy": "Lead a team meeting or collaborate on a project",
    "Sad": "Work on solo tasks or take a short mental break",
    "Angry": "Take a mindfulness session or complete low-stress tasks",
    "Neutral": "Continue regular assigned tasks",
    "Surprise": "Engage in innovative or brainstorming tasks",
    "Fear": "Do familiar tasks for reassurance",
    "Disgust": "Take a short break and return with a refreshed mind"
}
# --- Encryption Setup ---
key = Fernet.generate_key()  # Store this key securely in production
cipher_suite = Fernet(key)

# --- MongoDB Setup ---
client = MongoClient("mongodb://localhost:27017")
db = client["employee_mood_tracker"]
mood_collection = db["mood_logs"]

def log_mood(employee_id, emotion):
    mood_collection.insert_one({
        "employee_id": employee_id,
        "emotion": emotion,
        "timestamp": datetime.now()
    })

# --- Stress Management Alerts ---
def check_stress_alerts(db, employee_id, stress_moods=["sad", "angry", "disgust"], days=3, threshold=0.6):
    mood_log = db["mood_logs"]
    start_time = datetime.now() - timedelta(days=days)
    cursor = mood_log.find({
        "employee_id": employee_id,
        "timestamp": {"$gte": start_time}
    })

    df = pd.DataFrame(list(cursor))
    if df.empty:
        print(f"No recent mood data for {employee_id}.")
        return

    df['mood'] = df['emotion'].str.lower()
    stress_count = df['mood'].isin(stress_moods).sum()
    ratio = stress_count / len(df)

    if ratio >= threshold:
        print(f"\n🔴 ALERT: {employee_id} has shown high stress levels ({ratio*100:.1f}%) over the last {days} days.")
    else:
        print(f"\n🟢 {employee_id} shows normal mood trends.")

# --- Team Mood Analytics ---
def team_mood_analytics(db):
    mood_log = db["mood_logs"]
    cursor = mood_log.find()
    df = pd.DataFrame(list(cursor))

    if df.empty:
        print("No mood data available.")
        return

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['mood'] = df['emotion'].str.lower()

    mood_summary = df.groupby(['employee_id', pd.Grouper(key='timestamp', freq='D')])['mood'] \
                     .agg(lambda x: x.mode()[0] if not x.empty else None).reset_index()

    mood_counts = mood_summary.groupby('mood').size().reset_index(name='count')

    print("\n📊 Team Mood Distribution:")
    print(mood_counts)

    try:
        import matplotlib.pyplot as plt
        mood_counts.plot(kind='bar', x='mood', y='count', legend=False, title="Team Mood Summary")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("Install matplotlib to view charts.")
# --- Helper Functions ---
def anonymize_employee_id(employee_id):
    return hashlib.sha256(employee_id.encode()).hexdigest()

def encrypt_mood_data(mood):
    return cipher_suite.encrypt(mood.encode())

def decrypt_mood_data(encrypted_mood):
    return cipher_suite.decrypt(encrypted_mood).decode()

# --- Real-Time Emotion Detection with Task Recommendation ---
def run_emotion_recognition():
    cap = cv2.VideoCapture(0)
    emp_id = input("Enter employee ID: ")
    last_logged_time = datetime.min
    LOG_INTERVAL = timedelta(seconds=10)
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        for i, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                preds = classifier.predict(roi)[0]
                emotion = class_labels[preds.argmax()]
                task = TASK_MAP.get(emotion, "Keep working on current tasks")
                if datetime.now() - last_logged_time > LOG_INTERVAL:
                    log_mood(emp_id, emotion)
                    last_logged_time = datetime.now()
                cv2.putText(frame, f"Emotion: {emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(frame, f"Task: {task}", (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                cv2.putText(frame, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        cv2.imshow('Emotion Detector with Task Suggestion', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# --- Run All ---
if __name__ == "__main__":
    run_emotion_recognition()
    employee_id = input("\nEnter employee ID to check stress alert: ")
    check_stress_alerts(db, employee_id)
    print("\n--- TEAM MOOD ANALYTICS ---")
    team_mood_analytics(db)
    
