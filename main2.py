import csv
import copy
import cv2 as cv
import mediapipe as mp
from model import KeyPointClassifier
from app_files import calc_landmark_list, draw_info_text, draw_landmarks, get_args, pre_process_landmark
import pyttsx3

# Initialize the text-to-speech engine
engine = pyttsx3.init()

def vocalize_word(word):
    # Use the text-to-speech engine to vocalize the entire word
    engine.say(f'{word}')
    engine.runAndWait()

def main():
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]

    predicted_word = []  # List to store predicted letters and form words

    while True:
        key = cv.waitKey(10)
        if key == 27:  # ESC key (27) is pressed to exit the loop.
            break

        ret, image = cap.read()
        if not ret:
            break

        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                predicted_letter = keypoint_classifier_labels[hand_sign_id]

                predicted_word.append(predicted_letter)

                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    handedness,
                    ''.join(predicted_word)  # Display the predicted word
                )

        cv.imshow('Hand Gesture Recognition', debug_image)

        # Check if the space key is pressed to signal the end of a word
        if key == 32:  # Space key (32)
            word_to_vocalize = ''.join(predicted_word)
            vocalize_word(word_to_vocalize)
            predicted_word = []  # Clear the list for the next word

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
