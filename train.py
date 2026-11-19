import os
import json
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
DATASET_DIR = "dataset"
IMG_SIZE = 128
EPOCHS = 10
# 1. Load images (one subfolder per disease class)
class_names = sorted(os.listdir(DATASET_DIR))
images, labels = [], []
for index, name in enumerate(class_names):
    folder = os.path.join(DATASET_DIR, name)
    for file in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, file))
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        labels.append(index)
X = np.array(images)
y = np.array(labels)

# 2. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# 3. Build a simple CNN
model = keras.Sequential([
    layers.Rescaling(1.0 / 255, input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation="softmax"),
])
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
# 4. Train
model.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_test, y_test))
# 5. Save the model and class names (predict.py needs both)
model.save("model.keras")
with open("class_names.json", "w") as f:
    json.dump(class_names, f)
print("Saved model.keras and class_names.json")
