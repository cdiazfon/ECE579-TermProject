import pickle
import numpy as np

# Load file
with open('data8.pickle','rb') as file:
    data = pickle.load(file)

# Check data
print(f"Type of data: {type(data)}") # type dictionary

if isinstance(data, dict):
    print("Keys:", data.keys())
    print("\n")

#Extract data
x_train = data['x_train']
y_train = data['y_train']
x_validation = data['x_validation']
y_validation = data['y_validation']
x_test = data['x_test']
y_test = data['y_test']
labels = data['labels']

print(f"x_train shape: {x_train.shape}")
print(f"y_train labels shape: {y_train.shape}")
print(f"x_validation shape: {x_validation.shape}")
print(f"y_validation labels shape: {y_validation.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_test labels shape: {y_test.shape}")

print(f'Number of Labels: {len(labels)}')

