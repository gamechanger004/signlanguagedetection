import csv
import copy
import cv2 as cv
import mediapipe as mp
from model import KeyPointClassifier  # Import the custom KeyPointClassifier model.
from app_files import calc_landmark_list, draw_info_text, draw_landmarks, get_args, pre_process_landmark

def main():
    args = get_args()  # Parse command-line arguments.

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    # Create a VideoCapture object for the specified camera device and set its properties.
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

    keypoint_classifier = KeyPointClassifier()  # Initialize the custom KeyPointClassifier model.

    # Load keypoint classifier labels from a CSV file.
    with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]

    while True:
        key = cv.waitKey(10)
        if key == 27:  # ESC key (27) is pressed to exit the loop.
            break

        ret, image = cap.read()  # Read a frame from the camera.
        if not ret:
            break

        image = cv.flip(image, 1)  # Flip the frame horizontally.
        debug_image = copy.deepcopy(image)  # Create a deep copy of the frame for debugging.

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)  # Convert the frame to RGB color format.

        image.flags.writeable = False
        results = hands.process(image)  # Process the frame with the MediaPipe hands model.
        image.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                # Classify the hand gesture using the KeyPointClassifier.
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                debug_image = draw_landmarks(debug_image, landmark_list)  # Draw landmarks on the frame.

                debug_image = draw_info_text(
                    debug_image,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id]
                )  # Display information text about the recognized gesture.

        cv.imshow('Hand Gesture Recognition', debug_image)  # Display the processed frame.

    cap.release()  # Release the video capture object.
    cv.destroyAllWindows()  # Close all OpenCV windows.

if __name__ == '__main__':
    main()  # Execute the main function when the script is run.
