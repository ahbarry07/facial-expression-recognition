from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

from train import load_data
from preprocess import process_pixels

TEST_PATH = "./data/test.csv"

def predict():
    """
    Loads the trained model and evaluates its performance on the test set.
    """
        
    testX, testY = load_data(TEST_PATH)
    test_norm = process_pixels(testX)
    testY = to_categorical(testY, num_classes=7) # convert output class to category 

    model = load_model("./model/final_emotion_model.keras")

    _, test_acc = model.evaluate(test_norm, testY)

    print(f"\n\nAccuracy on test set: {test_acc * 100:.2f}%\n")
    
    

if __name__ == "__main__":
    predict()