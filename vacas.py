#uma exxplicação do codigo por emquanto já que não tem porra nenhm
from keras.models import load_model  # Imports the load_model function from Keras to load a pre-trained model.
import cv2  # Imports the opencv-python library, aliased as cv2.
import numpy as np # Imports the numpy library, commonly used for numerical operations, especially with arrays.
# Disable scientific notation for clarity
np.set_printoptions(suppress=True) # Configures numpy to suppress scientific notation when printing arrays.
# Load the model
model = load_model("keras_Model.h5", compile=False) # Loads the pre-trained Keras model from the file "keras_Model.h5". compile=False means the model will not be recompiled after loading.
# Load the labels
class_names = open("labels.txt", "r").readlines() # Opens the labels.txt file, reads all lines, and stores them in the class_names list. Each line is a class label.
# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0) # Initializes a video capture object to access the computer's default camera (usually camera 0).
while True: # Starts an infinite loop to continuously process frames from the camera.
    # Grab the webcamera's image.
    ret, image = camera.read() # Reads a frame from the camera. 'ret' is a boolean indicating success, and 'image' is the frame itself.
    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA) # Resizes the captured image to 224x224 pixels, which is likely the expected input size for the model.
    # Show the image in a window
    cv2.imshow("Webcam Image", image) # Displays the resized image in a window titled "Webcam Image".
    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3) # Converts the image to a numpy array of type float32 and reshapes it to the model's input shape (batch size of 1, 224x224 pixels, 3 color channels).
    # Normalize the image array
    image = (image / 127.5) - 1 # Normalizes the pixel values of the image to be within the range of -1 to 1, a common preprocessing step for neural networks.
    # Predicts the model
    prediction = model.predict(image) # Uses the loaded model to make a prediction on the processed image.
    index = np.argmax(prediction) # Gets the index of the class with the highest prediction score.
    class_name = class_names[index] # Retrieves the class name corresponding to the predicted index from the class_names list.
    confidence_score = prediction[0][index] # Gets the confidence score for the predicted class.
    # Print prediction and confidence score
    print("é o ", class_name[2:], end="") # Prints the predicted class name (skipping the first two characters, likely "0 " or "1 ").
    print("com ", str(np.round(confidence_score * 100))[:-2], "%"," de certeza") # Prints the confidence score as a percentage, rounded to the nearest whole number.
    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1) # Waits for a key press for 1 millisecond.
    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27: # Checks if the pressed key is the Escape key (ASCII 27).
        break # If the Escape key is pressed, the loop breaks.
camera.release() # Releases the camera resource.
cv2.destroyAllWindows() # Closes all OpenCV windows.