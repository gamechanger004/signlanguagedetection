import cv2 as cv
import mediapipe as mp
import pyttsx3
from model import KeyPointClassifier
from app_files import calc_landmark_list, draw_info_text, draw_landmarks, pre_process_landmark

# Initialize the text-to-speech engine
engine = pyttsx3.init()

def vocalize_letter(letter):
    # Use the text-to-speech engine to vocalize the predicted letter
    engine.say(f'{letter}')
    engine.runAndWait()

def sign_language_detection(frame, video_writer):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    keypoint_classifier = KeyPointClassifier()

    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    predicted_letter = None  # Initialize predicted_letter variable

    if results.multi_hand_landmarks is not None:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            landmark_list = calc_landmark_list(frame, hand_landmarks)
            pre_processed_landmark_list = pre_process_landmark(landmark_list)

            # Classify the hand gesture using the KeyPointClassifier.
            hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
            predicted_letter = chr(ord('A') + hand_sign_id)

            frame = draw_landmarks(frame, landmark_list)  # Draw landmarks on the frame.

            frame = draw_info_text(
                frame,
                handedness,
                predicted_letter
            )  # Display information text about the recognized gesture.

            # Vocalize the predicted letter
            vocalize_letter(predicted_letter)

    # Write the frame into the video stream
    video_writer.write(frame)

    return predicted_letter  # Return predicted_letter

def main():
    cap = cv.VideoCapture(0)  # Open the webcam

    if not cap.isOpened():
        print("Error: Failed to open webcam.")
        return

    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

    while cap.isOpened():  # Loop until the video stream ends
        ret, frame = cap.read()  # Read a frame from the webcam

        if not ret:
            print("Error: Failed to capture frame from webcam.")
            break

        # Run sign language detection on the frame
        predicted_letter = sign_language_detection(frame, out)

    # Release everything if job is finished
    cap.release()
    out.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
