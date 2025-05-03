from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPool2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras import Sequential

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import datetime

from preprocess import process_pixels
from plot import summarize_diagnostics

# Constant
TRAIN_PATH = "./data/train.csv"
INPUT_SHAPE = (48, 48, 1)
OUTPUT_CLASS = 7

initial_learning_rate = 0.0001


def load_data(path: str):
    """
    Load training or test dataset from a CSV file and return features and labels.

    This function reads a CSV file containing image pixel data and corresponding emotion labels,
    extracting the pixel values and labels into separate arrays for further processing.

    Args:
        path (str): Path to the CSV file (e.g., './data/train.csv') containing columns 'pixels'
                    and 'emotion'.

    Returns:
        tuple: A tuple (X, Y) where:
            - X (pandas.Series): Series of pixel strings from the 'pixels' column.
            - Y (pandas.Series): Series of integer labels from the 'emotion' column.

    Raises:
        FileNotFoundError: If the file at `path` does not exist.
        pd.errors.EmptyDataError: If the CSV file is empty.
        KeyError: If the required columns ('pixels', 'emotion') are missing from the CSV.

    Notes:
        - Requires `pandas` to be imported as `pd` in the global scope.
        - Assumes the CSV has 'pixels' as a space-separated string of 2304 values (48x48)
          and 'emotion' as integer labels (0 to 6).
    """
    
    df = pd.read_csv(path)
    X, Y = df["pixels"], df["emotion"]

    return X, Y



def build_model(input_shape: tuple, output_class: np.uint8):
    """
    Builds and compiles a deep CNN for image classification.

    Constructs a Sequential CNN model with multiple convolutional layers, batch normalization,
    max pooling, dropout for regularization, and fully connected layers. The model is compiled
    with the Adam optimizer and categorical crossentropy loss for multi-class classification.

    Args:
        input_shape (tuple): Shape of the input images (e.g., (48, 48, 1) for 48x48 grayscale).
        output_class (np.uint8): Number of output classes (e.g., 7 for 7 emotions).

    Returns:
        tensorflow.keras.Model: Compiled CNN model ready for training.

    Raises:
        ValueError: If `input_shape` is invalid or `output_class` is less than or equal to 0.
        tf.errors.InvalidArgumentError: If the model architecture fails to compile due to
                                        incompatible layer configurations.

    Notes:
        - Uses `Conv2D`, `BatchNormalization`, `MaxPool2D`, `Dropout`, `Flatten`, and `Dense`
          layers from TensorFlow/Keras.
        - Applies L2 regularization (0.01) to the last three convolutional layers.
        - Dropout rates increase from 0.25 to 0.4 in deeper layers to prevent overfitting.
        - Compiled with Adam optimizer (learning rate 0.0001) and 'categorical_crossentropy' loss.
        - Requires `tensorflow.keras` layers and `Adam` optimizer to be imported.
    """
     
    model= Sequential()
    
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape, name='conv_1_1'))
    model.add(Conv2D(64,(3,3), padding='same', activation='relu', name='conv_1_2'))
    model.add(BatchNormalization(name='batch_normalization_1'))
    model.add(MaxPool2D(pool_size=(2, 2), name='pool_1'))
    model.add(Dropout(0.25, name='dropout_1'))
    
    model.add(Conv2D(128,(5,5), padding='same', activation='relu', name='conv_2_1'))
    model.add(BatchNormalization(name='batch_normalization_2'))
    model.add(MaxPool2D(pool_size=(2, 2), name='pool_2'))
    model.add(Dropout(0.25, name='dropout_2'))
      
    model.add(Conv2D(256,(3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01), name='conv_3'))
    model.add(BatchNormalization(name='batch_normalization_3'))
    model.add(MaxPool2D(pool_size=(2, 2), name='pool_3'))
    model.add(Dropout(0.25, name='dropout_3'))
    
    model.add(Conv2D(512,(3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01), name='conv_4'))
    model.add(BatchNormalization(name='batch_normalization_4'))
    model.add(MaxPool2D(pool_size=(2, 2), name='pool_4'))
    model.add(Dropout(0.25, name='dropout_4'))
    
    model.add(Conv2D(512,(3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01), name='conv_5'))
    model.add(BatchNormalization(name='batch_normalization_5'))
    model.add(MaxPool2D(pool_size=(2, 2), name='pool_5'))
    model.add(Dropout(0.35, name='dropout_5'))
    
    model.add(Flatten()) 
    model.add(Dense(512,activation = 'relu', name='FCL_1'))
    model.add(BatchNormalization(name='batch_normalization_6'))
    model.add(Dropout(0.35, name='dropout_6'))
      
    model.add(Dense(1024,activation = 'relu', name='FCL_2'))
    model.add(BatchNormalization(name='batch_normalization_7'))
    model.add(Dropout(0.4, name='dropout_7'))
    
    model.add(Dense(output_class, activation='softmax'))
    
    model.compile(optimizer = Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model



def train(input_shape: tuple, output_class: np.uint8):
    """
    Trains the CNN model on the emotion dataset.

    Loads and preprocesses the training data, builds a CNN model, applies data augmentation,
    and trains the model with early stopping, learning rate reduction, and TensorBoard logging.
    Saves the trained model and generates diagnostic plots of training history.

    Args:
        input_shape (tuple): Shape of the input images (e.g., (48, 48, 1) for 48x48 grayscale).
        output_class (np.uint8): Number of output classes (e.g., 7 for 7 emotions).

    Returns:
        None: The function saves the model and plots but does not return a value.

    Raises:
        FileNotFoundError: If the training data file at `TRAIN_PATH` or the save directories
                          ('./model/', './results/') are inaccessible.
        ValueError: If the data preprocessing or model training fails due to invalid shapes
                    or configurations.
        tf.errors.InvalidArgumentError: If the model compilation or training fails due to
                                       incompatible data or model issues.
        AttributeError: If required functions (`load_data`, `process_pixels`, `build_model`,
                        `summarize_diagnostics`) or constants (`TRAIN_PATH`) are not defined.

    Notes:
        - Uses `ImageDataGenerator` for data augmentation with shifts, flips, and zooms.
        - Implements `EarlyStopping` (patience=5) and `ReduceLROnPlateau` (factor=0.5, min_lr=1e-6)
          to optimize training.
        - Logs training with TensorBoard to './results/logs/fit/YYYYMMDD-HHMMSS'.
        - Saves the model architecture to './model/final_emotion_model_arch.txt' and the model
          to './model/final_emotion_model.keras'.
        - Generates learning curves using `summarize_diagnostics` and saves them to
          './results/learn_and_loss_curves.png'.
        - Requires `tensorflow.keras`, `sklearn`, `pandas`, `numpy`, and `datetime` to be imported.
    """

    # load data
    trainX, trainY = load_data(TRAIN_PATH)
    
    # process data
    train_norm = process_pixels(trainX)
    
    trainY = to_categorical(trainY, num_classes=output_class)
    X_train, X_val, y_train, y_val = train_test_split(train_norm, trainY, test_size=0.3, random_state=43, stratify=trainY)
    
    # create data generator
    datagen = ImageDataGenerator(
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        horizontal_flip = True,
        zoom_range = 0.2
    )
    datagen.fit(train_norm)
    it_train = datagen.flow(X_train, y_train, batch_size=64)
    
    valgen = ImageDataGenerator(
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        horizontal_flip = True,
        zoom_range = 0.2
    )
    valgen.fit(X_val)
    it_val = datagen.flow(X_val, y_val, batch_size=64)
    
    # build model
    model = build_model(input_shape, output_class)
    with open('./model/final_emotion_model_arch.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    # tensorboard for monitor the learning process
    log_dir = "./results/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Early stopping for handle overfit
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # To slow down learning when the model stagnates, to try to improve convergence.
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    
    # fit model
    history = model.fit(it_train, validation_data=it_val, epochs=60, batch_size=100, callbacks=[tensorboard, early_stop, reduce_lr])
    
    # saving model
    model.save("./model/final_emotion_model.keras")
    
    # learning curves
    summarize_diagnostics(history)


if __name__ == "__main__":
    train(INPUT_SHAPE, OUTPUT_CLASS)