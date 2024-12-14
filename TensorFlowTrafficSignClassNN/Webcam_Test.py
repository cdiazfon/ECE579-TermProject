import cv2
import pickle
import numpy as np
import pandas as pd
from skimage import exposure
from tensorflow.keras.models import load_model

# Load train
with open("data8.pickle", "rb") as file:
    data = pickle.load(file)

# Load CVS
label_data = pd.read_csv("label_names.csv")
labels = dict(zip(label_data["ClassId"], label_data["SignName"]))
print(labels)

x_train = data['x_train']

mean = x_train.mean()
std = x_train.std()
print(f"Mean: {mean}, STD: {std}")

#traffic sign model
model = load_model("traffic_sign_CNN.keras")


def preprocess_frame(frame1):
    # Resize the frame
    resized_frame = cv2.resize(frame1, (32, 32))
    # Grayscale
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    # Histogram Equalization
    equalized_frame = exposure.equalize_adapthist(gray_frame, clip_limit=0.03)
    # Normalize
    normalized_frame = equalized_frame / 255.0
    # Standardized
    standardized_frame = (normalized_frame - mean) / std
    # add dimensions
    input_frame = np.expand_dims(standardized_frame, axis=(0, -1))
    return input_frame

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error capturing frame.")
        break

    input_frame = preprocess_frame(frame)

    # Predict
    predictions = model.predict(input_frame)
    predicted_class = np.argmax(predictions)
    predicted_label = labels.get(predicted_class, "Unknown")

    # Display the result
    cv2.putText(frame, f"Detected: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 158, 0), 2)

    cv2.imshow("Traffic Sign Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()