import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from src.data_preprocessing import create_dataset
from src.utils import set_seed
from models.cnn_lstm_model import create_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

set_seed(27)

features, labels, video_files_paths = create_dataset()

np.save(os.path.join(ARTIFACTS_DIR, "features.npy"), features)
np.save(os.path.join(ARTIFACTS_DIR, "labels.npy"), labels)
np.save(os.path.join(ARTIFACTS_DIR, "video_files_paths.npy"), video_files_paths)

one_hot_encoded_labels = to_categorical(labels)

X_train, X_test, Y_train, Y_test = train_test_split(
    features,
    one_hot_encoded_labels,
    test_size=0.1,
    shuffle=True,
    random_state=27
)

model = create_model()
model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.6,
    patience=5,
    min_lr=0.00005,
    verbose=1
)

history = model.fit(
    x=X_train,
    y=Y_train,
    epochs=50,
    batch_size=24,
    shuffle=True,
    validation_split=0.2
)

model.save(os.path.join(MODELS_DIR, "MobBiLSTM_model.keras"))
print("Model saved successfully.")
