import cv2
import speech_recognition as sr
from transformers import pipeline
from deepface import DeepFace

# Load sentiment analyzer
text_emotion_analyzer = pipeline("sentiment-analysis")

# --- Text Emotion Detection ---
def detect_text_emotion(text):
    result = text_emotion_analyzer(text)
    return result[0]['label']

# --- Speech Emotion Detection ---
def detect_speech_emotion():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        print("You said:", text)
        return detect_text_emotion(text)
    except sr.UnknownValueError:
        return "Neutral"
    except sr.RequestError:
        return "Error"



# --- Combined Emotion Detection ---
def real_time_emotion_detection():

    print("Detecting speech emotion...")
    speech_emotion = detect_speech_emotion()
    print("Speech Emotion:", speech_emotion)

    print("Enter a message for text emotion detection:")
    user_text = input("Text: ")
    text_emotion = detect_text_emotion(user_text)
    print("Text Emotion:", text_emotion)

    return {
        "speech_emotion": speech_emotion,
        "text_emotion": text_emotion
    }

# Run the function
if __name__ == "__main__":
    emotions = real_time_emotion_detection()
    print("\nDetected Emotions:", emotions)
