import numpy as np
import tensorflow as tf

class KeyPointClassifier(object):
    def __init__(
        self,
        model_path='model/keypoint_classifier/keypoint_classifier.tflite',
        num_threads=1,
    ):
        # Initialize the KeyPointClassifier object with the path to a TFLite model and the number of threads for inference.
        self.interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)

        # Allocate memory for the TFLite model and get input/output details.
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(
        self,
        landmark_list,
    ):
        # Define a callable method for the KeyPointClassifier class that performs classification.

        # Get the index of the input tensor in the TFLite model details.
        input_details_tensor_index = self.input_details[0]['index']

        # Set the input tensor to the provided landmark list as a NumPy array of float32.
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([landmark_list], dtype=np.float32))

        # Run inference on the TFLite model.
        self.interpreter.invoke()

        # Get the index of the output tensor in the TFLite model details.
        output_details_tensor_index = self.output_details[0]['index']

        # Get the output tensor from the TFLite model.
        result = self.interpreter.get_tensor(output_details_tensor_index)

        # Find the index of the maximum value in the result tensor (classification result).
        result_index = np.argmax(np.squeeze(result))

        # Return the index of the predicted class or keypoint.
        return result_index
