from datetime import datetime
from tensorflow import keras
import numpy as np
import time
import cv2

from preprocess import preprocess_frame


# Load the pre-trained DNN face detection model
caffeModel = "./model/res10_300x300_ssd_iter_140000.caffemodel"
prototxtPath = "./model/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(prototxtPath, caffeModel)

# Parameters
FACE_CACHE_DURATION = 0.5  # Seconds to reuse last face position for rectangle
CONFIDENCE_THRESHOLD = 0.5  # Confidence threshold for DNN face detection
IMG_SIZE = 48  # CNN input size (48x48)

# Load the trained CNN model
model = keras.models.load_model('./model/final_emotion_model.keras')

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']



def predict_emotion(model, image, emotion_labels, isFrame=False):
    """
    Predict the emotion from an image using a trained CNN model.

    This function processes an input image (either a raw frame or a preprocessed image) using
    a Convolutional Neural Network (CNN) model to predict the dominant emotion. The image is
    preprocessed if it is a raw frame, otherwise it is assumed to be in the correct format.
    The function returns the predicted emotion label and its associated probability.

    Args:
        model: Trained TensorFlow/Keras model for emotion prediction.
        image: Input image data. If `isFrame` is True, a raw BGR frame (numpy.ndarray);
               otherwise, a preprocessed array with shape (1, 48, 48, 1).
        emotion_labels (list): List of emotion labels (e.g., ['Angry', 'Disgust', ...]).
        isFrame (bool, optional): If True, indicates the input is a raw frame requiring
            preprocessing; if False, assumes the image is already preprocessed. Defaults to False.

    Returns:
        tuple: A tuple (emotion, probability) where:
            - emotion (str): The predicted emotion label from `emotion_labels`.
            - probability (float): The confidence score (0.0 to 1.0) of the prediction.

    Raises:
        ValueError: If the input image has an invalid shape or if `emotion_labels` is empty.
        tf.errors.InvalidArgumentError: If the model fails to process the input due to shape mismatch.

    Notes:
        - If `isFrame` is True, the image is preprocessed using `preprocess_frame` with `IMG_SIZE`.
        - The preprocessed image must have shape (1, 48, 48, 1) to match the model's input.
        - The function assumes `preprocess_frame` and `IMG_SIZE` are defined in the global scope.
    """

    if isFrame:
        img_processed = preprocess_frame(image, IMG_SIZE)
    else:
        img_processed = image.reshape(1, 48, 48, 1)

    predictions = model.predict(img_processed, verbose=0)[0]
    max_idx = np.argmax(predictions)
    return emotion_labels[max_idx], predictions[max_idx]



def detect_faces_dnn(frame, net, confidence_threshold=0.5):
    """
    Detect faces in an image using a pre-trained Deep Neural Network (DNN) model.

    This function processes an input image frame using a DNN-based face detection model
    (e.g., OpenCV's SSD model) to identify face locations. It creates a blob from the
    resized image, performs forward propagation, and extracts bounding box coordinates
    for faces with confidence scores above the specified threshold.

    Args:
        frame (numpy.ndarray): Input image frame in BGR format with shape (height, width, 3).
        net (cv2.dnn_Net): Pre-trained DNN model (e.g., loaded with cv2.dnn.readNetFromCaffe).
        confidence_threshold (float, optional): Minimum confidence score (0.0 to 1.0) for
            accepting a detection. Defaults to 0.5.

    Returns:
        list: List of tuples (x, y, w, h) representing the top-left corner coordinates (x, y)
              and dimensions (width w, height h) of detected faces. Returns an empty list if
              no faces are detected.

    Raises:
        ValueError: If the input frame is None or has invalid dimensions.
        cv2.error: If the DNN model fails to process the input (e.g., due to invalid net or blob).

    Notes:
        - The image is resized to 300x300 pixels for processing.
        - Mean subtraction values (104.0, 177.0, 123.0) are applied to normalize the input.
        - Bounding box coordinates are clipped to ensure they stay within the frame boundaries.
        - Only boxes with positive width and height are included in the result.
    """
    
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x2, y2) = box.astype("int")
            x = max(0, x)
            y = max(0, y)
            x2 = min(w, x2)
            y2 = min(h, y2)
            w_box = x2 - x
            h_box = y2 - y
            if w_box > 0 and h_box > 0:
                faces.append((x, y, w_box, h_box))
    return faces


def process_video_stream():
    """
    Process video stream, detect all faces with DNN, annotate with persistent rectangles, and predict emotions in real-time.

    This function captures a video stream from a webcam, detects faces using a pre-trained Deep Neural Network (DNN),
    annotates detected faces with persistent rectangles, and predicts emotions using a trained CNN model. Emotions are
    updated every second and displayed on the frame with probabilities. The system supports multiple face tracking
    with a cache duration to maintain annotations temporarily after face detection.

    Args:
        None: The function relies on global variables: `cv2`, `net` (DNN model), `model` (CNN model),
              `emotion_labels` (list of emotion labels), `CONFIDENCE_THRESHOLD` (float), `FACE_CACHE_DURATION` (float),
              `predict_emotion` (function), `detect_faces_dnn` (function), `datetime`, and `time`.

    Returns:
        None: The function modifies the display and prints results but does not return a value.

    Raises:
        cv2.error: If the webcam fails to initialize or read frames.
        ValueError: If the DNN or CNN model processing fails due to invalid input or configuration.
        AttributeError: If required global variables (e.g., `net`, `model`) are not defined.

    Notes:
        - The webcam is accessed with `cv2.VideoCapture(0)`; change the index if multiple cameras are present.
        - Faces are detected using `detect_faces_dnn` with a configurable confidence threshold.
        - Emotions are predicted using `predict_emotion` every second, with results cached for `FACE_CACHE_DURATION` seconds.
        - The function displays the output in a window titled 'Video Stream' and exits with the 'q' key.
        - The frame dimensions are retrieved from the webcam properties for coordinate validation.
        - Error handling includes skipping invalid face predictions and stopping on frame read failures.
    """
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Webcam not available please change the value in the 'VideoCapture' object")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    last_prediction_time = time.time()
    last_faces = []  # Store last face coordinates [(x, y, w, h), ...]
    last_face_time = time.time()  # Time of last face detection
    emotions = []  # Store emotions for each face
    probabilities = []  # Store probabilities for each face

    print("\nReading video stream ...\n")

    while True:
        current_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame.")
            break

        # Detect faces using DNN
        faces = detect_faces_dnn(frame, net, CONFIDENCE_THRESHOLD)

        # Update last faces if new faces are detected
        if len(faces) > 0:
            last_faces = faces
            last_face_time = current_time
            # Resize emotions and probabilities to match number of faces
            emotions = [emotions[i] if i < len(emotions) else None for i in range(len(faces))]
            probabilities = [probabilities[i] if i < len(probabilities) else None for i in range(len(faces))]

        # Process all faces (current or cached)
        if last_faces and (current_time - last_face_time) < FACE_CACHE_DURATION:
            for i, (x, y, w, h) in enumerate(last_faces):
                # Validate coordinates
                if 0 <= x < frame_width and 0 <= y < frame_height and x+w <= frame_width and y+h <= frame_height:
                    face_img = frame[y:y+h, x:x+w]
                    # Predict emotion every second
                    if current_time - last_prediction_time >= 1:
                        try:
                            emotion, probability = predict_emotion(model, face_img, emotion_labels, isFrame=True)
                            emotions[i] = emotion
                            probabilities[i] = probability
                        except:
                            # Skip prediction if face_img is invalid
                            continue
                    # Draw rectangle and label for each face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    if emotions[i] and probabilities[i]:
                        label = f"{emotions[i]}: {probabilities[i]:.2f}"
                        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            # Reset face tracking if cache duration expires
            last_faces = []
            emotions = []
            probabilities = []

        # Update prediction time and print emotions
        if current_time - last_prediction_time >= 1:
            last_prediction_time = current_time
            if emotions and any(emotions):  # Print only if there are detected emotions
                current_time_str = datetime.now().strftime('%H:%M:%S')
                for i, emotion in enumerate(emotions):
                    if emotion and probabilities[i]:
                        print(f"{current_time_str}s : Face {i+1} - {emotion} , {int(probabilities[i] * 100)}%")

        # Display the frame
        cv2.imshow('Video Stream', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Stopped processing.")



if __name__ == "__main__":
    process_video_stream()