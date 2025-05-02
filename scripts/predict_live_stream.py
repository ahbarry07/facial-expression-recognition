from tensorflow import keras
import numpy as np
import time
import cv2
from datetime import datetime

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

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

def preprocess_frame(face_img):
    """Preprocess a face image to match CNN input (48x48 grayscale)."""

    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    normalized = resized / 255.0
    return normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)


def predict_emotion(face_img):
    """Predict emotion using the trained CNN."""

    preprocessed = preprocess_frame(face_img)
    predictions = model.predict(preprocessed, verbose=0)[0]
    max_idx = np.argmax(predictions)
    return emotion_labels[max_idx], predictions[max_idx]


def detect_faces_dnn(frame, net, confidence_threshold=0.5):
    """Detect faces in the frame using the DNN model."""
    
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
    """Process video stream, detect all faces with DNN, annotate with persistent rectangles, and predict emotions in real-time."""
    
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
                            emotion, probability = predict_emotion(face_img)
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