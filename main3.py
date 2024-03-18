import streamlit as st
import cv2 as cv
import mediapipe as mp
import pyttsx3
import numpy as np
from model import KeyPointClassifier
from app_files import calc_landmark_list, draw_info_text, draw_landmarks, pre_process_landmark

# Initialize the text-to-speech engine
engine = pyttsx3.init()

def vocalize_letter(letter):
    # Use the text-to-speech engine to vocalize the predicted letter
    engine.say(f'{letter}')
    engine.runAndWait()

def sign_language_detection(image):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    keypoint_classifier = KeyPointClassifier()

    results = hands.process(image)

    if results.multi_hand_landmarks is not None:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            landmark_list = calc_landmark_list(image, hand_landmarks)
            pre_processed_landmark_list = pre_process_landmark(landmark_list)

            # Classify the hand gesture using the KeyPointClassifier.
            hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
            predicted_letter = chr(ord('A') + hand_sign_id)

            image = draw_landmarks(image, landmark_list)  # Draw landmarks on the frame.

            image = draw_info_text(
                image,
                handedness,
                predicted_letter
            )  # Display information text about the recognized gesture.

            # Vocalize the predicted letter
            vocalize_letter(predicted_letter)

    return image

def main():
    st.title("Sign Language Detection Web App")

    cap = cv.VideoCapture(0)  # Open the webcam

    while True:
        _, frame = cap.read()  # Read a frame from the webcam

        # Convert the frame to RGB color format
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Run sign language detection on the frame
        processed_frame = sign_language_detection(rgb_frame)

        # Display the processed frame
        st.image(processed_frame, channels="RGB", use_column_width=True)

if __name__ == '__main__':
    main()
