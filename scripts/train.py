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


# load train and test dataset
def load_data(path: str):
    """This fonction load data and return X and Y"""
    
    df = pd.read_csv(path)
    X, Y = df["pixels"], df["emotion"]

    return X, Y



# Build model
def build_model(input_shape: tuple, output_class: np.uint8):
    """
    Builds and compiles a deep CNN for image classification.
    Includes convolutional layers, batch normalization, pooling, dropout, and fully connected layers.
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



#Train CNN
def train(input_shape: tuple, output_class: np.uint8):
    """
    Trains the CNN model on the emotion dataset.
    Includes data augmentation, early stopping, learning rate reduction, TensorBoard logging, 
    and saves the trained model and training history plots.
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