from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

from preprocess import process_pixels
from train import load_data


TEST_PATH = "./data/test.csv"

def predict():
    """
    Loads the trained model and evaluates its performance on the test set.

    This function loads a pre-trained Keras model and assesses its accuracy on a test dataset.
    It processes the test images by normalizing pixel values and converting labels to categorical
    format, then computes and prints the test set accuracy.

    Args:
        None: The function relies on global variables: `TEST_PATH` (string path to test data),
              `load_data` (function to load data), `process_pixels` (function to normalize pixels),
              `to_categorical` (function to convert labels), and `load_model` (function to load the model).

    Returns:
        None: The function prints the test accuracy but does not return a value.

    
    Notes:
        - The test data is loaded using `load_data` with the path defined by `TEST_PATH`.
        - Pixel values are normalized using `process_pixels` before evaluation.
        - Labels are converted to a categorical format with 7 classes using `to_categorical`.
        - The model is loaded from './model/final_emotion_model.keras' and evaluated on the normalized test set.
        - Accuracy is printed as a percentage with two decimal places.
        - Requires `tensorflow` and `numpy` to be imported in the global scope.
    """
        
    testX, testY = load_data(TEST_PATH)
    test_norm = process_pixels(testX)
    testY = to_categorical(testY, num_classes=7) # convert output class to category 

    model = load_model("./model/final_emotion_model.keras")

    _, test_acc = model.evaluate(test_norm, testY)

    print(f"\n\nAccuracy on test set: {test_acc * 100:.2f}%\n")
    
    

if __name__ == "__main__":
    predict()