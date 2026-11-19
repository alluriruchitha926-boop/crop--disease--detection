import sys
import json
import numpy as np
import cv2
from tensorflow import keras
from severity import calculate_severity
IMG_SIZE = 128
def predict_disease(image_path):
    model = keras.models.load_model("model.keras")
    # Class names saved by train.py
    with open("class_names.json") as f:
        class_names = json.load(f)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image: " + image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)  # add batch dimension

    probs = model.predict(img)[0]
    index = int(np.argmax(probs))
    return class_names[index], float(probs[index]) * 100
if __name__ == "__main__":
    path = sys.argv[1]
    disease, confidence = predict_disease(path)
    percent, level = calculate_severity(path)
    print(f"Disease:    {disease} ({confidence:.1f}% confidence)")
    print(f"Diseased area: {percent}%")
    print(f"Severity:   {level}")
