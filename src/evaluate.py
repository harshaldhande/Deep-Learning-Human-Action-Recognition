import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "MobBiLSTM_model.keras")

CLASSES_LIST = ["Explosion", "Fighing", "Road_acc"]

features = np.load(os.path.join(ARTIFACTS_DIR, "features.npy"))
labels = np.load(os.path.join(ARTIFACTS_DIR, "labels.npy"))

from tensorflow.keras.utils import to_categorical
labels_one_hot = to_categorical(labels)

X_train, X_test, Y_train, Y_test = train_test_split(
    features,
    labels_one_hot,
    test_size=0.1,
    shuffle=True,
    random_state=27
)

model = load_model(MODEL_PATH)

Y_pred = model.predict(X_test)
Y_test_labels = np.argmax(Y_test, axis=1)
Y_pred_labels = np.argmax(Y_pred, axis=1)

accuracy = accuracy_score(Y_test_labels, Y_pred_labels)
precision = precision_score(Y_test_labels, Y_pred_labels, average="weighted")
recall = recall_score(Y_test_labels, Y_pred_labels, average="weighted")

print("Classification Report:")
print(classification_report(Y_test_labels, Y_pred_labels, target_names=CLASSES_LIST))
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
