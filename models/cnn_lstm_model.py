from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, TimeDistributed, Dropout, Flatten, Dense
from .cnn_model import create_mobilenet
from .lstm_model import create_bidirectional_lstm

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
SEQUENCE_LENGTH = 20
CLASSES_LIST = ["Explosion", "Fighing", "Road_acc"]

def create_model():
    mobilenet = create_mobilenet(None)

    model = Sequential()
    model.add(Input(shape=(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
    model.add(TimeDistributed(mobilenet))
    model.add(Dropout(0.25))
    model.add(TimeDistributed(Flatten()))
    model.add(create_bidirectional_lstm())
    model.add(Dropout(0.25))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(len(CLASSES_LIST), activation="softmax"))
    return model
