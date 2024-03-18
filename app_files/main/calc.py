import numpy as np
import cv2 as cv

def calc_landmark_list(image, landmarks):
    # Get the width and height of the input image
    image_width, image_height = image.shape[1], image.shape[0]

    # Initialize an empty list to store landmark points
    landmark_point = []

    # Iterate over the detected landmarks
    for _, landmark in enumerate(landmarks.landmark):
        # Calculate the x-coordinate of the landmark in pixel values, ensuring it stays within the image width
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        
        # Calculate the y-coordinate of the landmark in pixel values, ensuring it stays within the image height
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        
        # Append the calculated landmark coordinates to the list
        landmark_point.append([landmark_x, landmark_y])
        
    # Return the list of landmark points
    return landmark_point
