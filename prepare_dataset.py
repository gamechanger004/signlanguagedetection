import copy
import cv2 as cv
import mediapipe as mp
from app_files import calc_landmark_list, draw_landmarks, get_args, pre_process_landmark, logging_csv

# Import necessary libraries/modules

def main():
    args = get_args()  # Get command-line arguments using a custom function

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    cap = cv.VideoCapture(cap_device)  # Open a video capture device based on the provided device number
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)  # Set the video frame width
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)  # Set the video frame height

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,  # Use static image mode if specified
        max_num_hands=1,  # Detect a maximum of 1 hand
        min_detection_confidence=min_detection_confidence,  # Minimum confidence for hand detection
        min_tracking_confidence=min_tracking_confidence,  # Minimum confidence for hand tracking
    )

    mode = 1
    number = -1

    while True:
        key = cv.waitKey(10)  # Wait for a key press with a 10 ms timeout
        if key == 27:  # If the ESC key (27) is pressed, exit the loop
            break

        if 48 <= key <= 57:  # Check if a number key (0-9) is pressed and store the number
            number = key - 48

        ret, image = cap.read()  # Read a frame from the video capture device

        if not ret:
            break

        image = cv.flip(image, 1)  # Flip the image horizontally (mirror effect)
        debug_image = copy.deepcopy(image)  # Create a deep copy of the image for debugging

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)  # Convert the image to RGB color format
        image.flags.writeable = False  # Make the image read-only to avoid data modification
        results = hands.process(image)  # Process the image to detect and track hands
        image.flags.writeable = True  # Make the image writable again

        if results.multi_hand_landmarks is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract landmark coordinates from the detected hand landmarks
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                # Pre-process the landmark data if needed
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                # Log the data along with the associated number and mode
                logging_csv(number, mode, pre_processed_landmark_list)
                # Draw landmarks on the debug image
                debug_image = draw_landmarks(debug_image, landmark_list)
                info_text = "Press key 0-9"  # Display an information text
                cv.putText(debug_image, info_text, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1.0, (196, 161, 33), 1, cv.LINE_AA)
                cv.imshow('Dataset Preparation', debug_image)  # Display the debug image

    cap.release()  # Release the video capture device
    cv.destroyAllWindows()  # Close all OpenCV windows

if __name__ == '__main__':
    main()  # Call the `main` function when the script is executed
