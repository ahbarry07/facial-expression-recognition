# facial-expression-recognition

## Project Overview

Welcome to the **Facial Expression Recognition** project! This project leverages computer vision and deep learning to detect and classify human emotions in real-time from a webcam video stream. Using a Convolutional Neural Network (CNN) built from scratch, the system identifies seven facial emotions: 😃 **Happy**, 😔 **Sad**, 😡 **Angry**, 😲 **Surprise**, 😱 **Fear**, 😖 **Disgust**, and 😐 **Neutral**. It also includes an adversarial attack to subtly modify an image, tricking the CNN into misclassifying emotions.

The project integrates **face tracking** with **emotion classification**, utilizing OpenCV for face detection and TensorFlow/Keras for CNN development. Additionally, it explores neural network vulnerabilities through an adversarial attack, demonstrating how slight image changes can alter predictions.

### Learning Objectives
- Implement and train CNNs for image classification.
- Process video streams with OpenCV for real-time face detection.
- Achieve >60% accuracy on the test set with a custom-trained CNN.
- Explore adversarial attacks to understand model limitations.

### Project Features
- **Emotion Classification**: A custom CNN classifies seven facial emotions from 48x48 grayscale images.
- **Real-Time Face Tracking**: Uses OpenCV's DNN model to track multiple faces live.
- **Real-Time Prediction**: Displays emotion predictions with probabilities every second.
- **Adversarial Attack**: Modifies a "Happy" image to be misclassified as "Sad."
- **Training Monitoring**: Integrates TensorBoard for visualizing training progress.

## Project Structure

The project is organized as follows:

```
facial-expression-recognition/
├── data/                  # CSV-based FER-2013 dataset
├── model/                 # Trained model + OpenCV face detector files
├── results/               # Visual results: original/hacked images, learning curves
|     └── logs/            # TensorBoard logs
├── scripts/               # Core Python scripts (training, live prediction, attack)
├── README.md              # This file
└── requirements.txt       # Python dependencies

```

## Installation

### Prerequisites
- Python 3.8 or higher
- A webcam (for real-time emotion detection)
- Git (optional, for cloning the repository)

### Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ahbarry07/facial-expression-recognition.git
   cd facial-expression-recognition
   ```

2. **Create a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
   - ***Virtual env with conda***

   ```bash
   conda create -n myenv python=3.12
   conda activate my_env
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Requirements are listed in `requirements.txt` (TensorFlow, OpenCV, NumPy, etc.).

4. **Verify Pre-trained Models and Data**:
    - ***Training Environment***:  Due to the need for GPU acceleration, the model was trained on **Google Colab**.
   - Ensure `final_emotion_model.keras`, `res10_300x300_ssd_iter_140000.caffemodel`, and `deploy.prototxt.txt` are in the `model/` directory.
   - The dataset (`train.csv` and `test.csv`) should be in the `data/` directory.

5. **Prepare Input Data** (for adversarial attack):
   - Place a sample image (`happy.png`) in the `results/` directory.

## 🎥 Demo

Watch the demonstration of the facial expression recognition system in action:

[![Watch the demo](https://img.youtube.com/vi/VIDEO_ID/0.jpg)]([https://www.youtube.com/watch?v=kgeh7Br6ORo])

Or download the video directly here: [demo.mp4](results/demo.mp4)


## Usage

### 1. Real-Time Emotion Detection
The `predict_live_stream.py` script processes a live webcam feed, detects multiple faces, and predicts emotions in real-time.

**Run the script**:
```bash
python scripts/predict_live_stream.py
```

**What to Expect**:
- A window displays the webcam feed with green rectangles around detected faces.
- Each face is labeled with the predicted emotion and probability (e.g., "Happy: 0.75").
- The console prints predictions every second for each face (e.g., "12:34:56 : Face 1 - Happy , 75%").
- Press `q` to stop the stream and exit.
- If the webcam fails, update the `cv2.VideoCapture(0)` line to use an alternative video source.

**Note**: This implementation does not save images or videos, focusing on real-time predictions.

### 2. Adversarial Attack
The `hack_cnn.py` script performs an adversarial attack on an image predicted as "Happy," modifying it to be classified as "Sad."

**Run the script**:
```bash
python scripts/hack_cnn.py
```

**What to Expect**:
- Loads `happy.png` from the `results/` directory.
- Verifies a "Happy" prediction with >90% probability.
- Applies an adversarial attack to achieve a "Sad" prediction.
- Logs iteration progress (e.g., "Iteration 1: Predicted Emotion: Happy, Probability: 95.1234").
- Saves images in `results/`:
  - `original_happy_image.png`
  - `modified_sad_image.png`
  - `comparison.png`
- Prints the final prediction (e.g., "Modified Prediction: Sad, Probability: 92.5678").

### 3. Training and Visualization
- **Training Script**: Use `scripts/train.py` to retrain the CNN (if needed) on `data/train.csv`.
- **Artifacts**:
  - `model/final_emotion_model_arch.txt`: Explains the CNN architecture (generated with `model.summary()`).
  - `results/learn_and_loss_curves.png`: Plots training and validation metrics, showing early stopping before overfitting.
- **Logs**: Training logs are stored in `logs/fit_20250428-141753/`.

### 4. Additional Scripts
- `scripts/preprocess.py`: Handles image preprocessing for CNN input.
- `scripts/predict_test_set.py`: Evaluates the model on the test set (`data/test.csv`).
- `scripts/plot.py`: Generates plots (e.g., learning curves).

## Implementation Details

### Emotion Classification
- **Dataset**: Trained on the FER-2013 dataset (`data/train.csv` and `data/test.csv`), achieving >60% accuracy.
- **Model**: A custom CNN built from scratch with TensorFlow/Keras, saved as `model/final_emotion_model.keras`.
- **Training**:
  - Used early stopping and TensorBoard for monitoring.
  - Documented in `model/final_emotion_model_arch.txt`.

### Face Tracking
- **Detection**: Uses OpenCV's DNN model for multi-face detection.
- **Real-Time**: Processes webcam feed, annotating all detected faces with emotions every second.

### Adversarial Attack
- Modifies a "Happy" image (>90% probability) to "Sad" using a gradient-based attack.
- Results saved in `results/` for comparison.

## Challenges and Solutions
- **Overfitting**: Mitigated with early stopping, tracked via TensorBoard.
- **Multiple Faces**: Ensured all faces are detected and annotated.
- **Performance**: Optimized for real-time prediction without saving data.

## Future Improvements
- Experiment with pre-trained CNNs for higher accuracy.
- Add a GUI for interactive emotion display.
- Test robustness across diverse conditions.
- Implement adversarial defenses.

## Acknowledgments
- **Andrew Ng's Course**: Foundation for CNN knowledge.
- **FER-2013 Dataset**: Provided by Kaggle.
- **OpenCV and TensorFlow**: Key tools for implementation.

## Contact
For questions or contributions, contact [barryahmadou135@gmail.com](barryahmadou135@gmail.com) or open an issue on the GitHub repository.

---

*Built with curiosity and a passion for computer vision!*
