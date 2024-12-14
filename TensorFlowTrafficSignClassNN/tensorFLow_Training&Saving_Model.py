import pickle
import numpy as np
import tensorflow as tf
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from itertools import combinations
import os
import pandas as pd
import random


# Create Model
def creates_cnn(input1, classes):
    model = Sequential()
    # Input Layer
    # model.add(Input(input1=input1))
    # CNN 1
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.2))
    # CNN 2
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.3))
    # CNN 3
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.4))
    # Flatten
    model.add(Flatten())
    # Fully connected layer
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    # Output Layer
    model.add(Dense(classes, activation='softmax'))
    return model


def main():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("GPU is available")
        for gpu in gpus:
            print(f"Device: {gpu.name}\n")
        #
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("No GPU, using CPU")

    # Load file
    with open('data0.pickle', 'rb') as file:
        data = pickle.load(file)

    # Check data type
    # print(f"Type of data: {type(data)}")
    # if isinstance(data, dict):
    #     print("Keys:", data.keys(), "\n")

    # Extract data form file
    x_train = data['x_train']
    y_train = data['y_train']
    x_validation = data['x_validation']
    y_validation = data['y_validation']
    x_test = data['x_test']
    y_test = data['y_test']
    labels = data['labels']

    folds = 10

    print("Splitting data into 10 folds...")
    x_train_partitions = np.array_split(x_train, folds)
    y_train_partitions = np.array_split(y_train, folds)
    x_valid_partitions = np.array_split(x_validation, folds)
    y_valid_partitions = np.array_split(y_validation, folds)
    x_test_partitions = np.array_split(x_test, folds)
    y_test_partitions = np.array_split(y_test, folds)

    all_histories = {}
    prev_val_indices = []

    # for fold in range(0, folds):
    print("Selecting 2 folds for testing, 1 for validation, and 7 for training...")
    for test_indices in combinations(range(10), 2):  # Select 2 folds for testing

        if test_indices == (1, 2):
            break

        # if test_indices == (0, 1):
        possible_val_indices = [i for i in range(10) if i not in test_indices and i not in prev_val_indices]

        # select random index for validation fold
        val_index = random.choice(possible_val_indices)

        # Training folds: All remaining folds except the validation fold
        possible_train_indices = [i for i in range(10) if i not in test_indices and i != val_index]
        train_indices = [i for i in possible_train_indices]

        # Remove the selected number from the list
        prev_val_indices.append(val_index)

        print(f"Current test indices: {test_indices}")
        print(f"Current val index: {val_index}")
        print(f"Current indices for concatenating train folds: {train_indices}")

        # Combine the folds for training
        train_x_data = np.concatenate([x_train_partitions[i] for i in train_indices])
        train_y_data = np.concatenate([y_train_partitions[i] for i in train_indices])

        val_x_data = x_valid_partitions[val_index]
        val_y_data = y_valid_partitions[val_index]

        test_x_data = np.concatenate([x_test_partitions[i] for i in test_indices])
        test_y_data = np.concatenate([y_test_partitions[i] for i in test_indices])

        # Print Shape
        # print(f"x_train shape: {train_x_data.shape}")
        # print(f"y_train labels shape: {train_y_data.shape}")
        # print(f"x_validation shape: {val_x_data.shape}")
        # print(f"y_validation labels shape: {val_y_data.shape}")
        # print(f"x_test shape: {test_x_data.shape}")
        # print(f"y_test labels shape: {test_y_data.shape}\n")

        # Transpose
        train_x_data = np.transpose(train_x_data, (0, 2, 3, 1))
        val_x_data = np.transpose(val_x_data, (0, 2, 3, 1))
        test_x_data = np.transpose(test_x_data, (0, 2, 3, 1))

        # Convert to one-hot
        classes_num = len(labels)
        val_y_data = to_categorical(val_y_data, classes_num)
        train_y_data = to_categorical(train_y_data, classes_num)
        test_y_data = to_categorical(test_y_data, classes_num)

        # Input shape
        input_system_shape = (32, 32, 3)

        # Create CNN model
        model = creates_cnn(input_system_shape, classes_num)

        num_epochs = 50

        # Prepare model for training
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Print summary
        #model.summary()

        # Train model
        print(f"Training for test folds: {test_indices}")
        start_time = time.time()
        train_model = model.fit(
            train_x_data, train_y_data,
            validation_data=(val_x_data, val_y_data),
            epochs=num_epochs,
            batch_size=128
        )

        # Save the history for the combination
        history_key = f"test_{test_indices}"
        all_histories[history_key] = train_model.history

        end_time = time.time()
        print("Total time elapsed: %.2f" % (end_time - start_time))
        print(f"Training complete for test folds: {test_indices} ! Saving model...")
        test_loss, test_accuracy = model.evaluate(test_x_data, test_y_data)
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

        # Plot training & validation loss
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_model.history['loss'], label='Train Loss')
        plt.plot(train_model.history['val_loss'], label='Validation Loss')
        plt.title(f'Loss for test folds: {test_indices}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'loss_testfolds_{test_indices}_{num_epochs}.png')

        # Plot training & validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot(train_model.history['accuracy'], label='Train Accuracy')
        plt.plot(train_model.history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'Accuracy for test folds: {test_indices}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(f'accuracy_testfolds_{test_indices}_{num_epochs}.png')

        plt.tight_layout()
        plt.show()

    # Create the directory if it doesn't exist
    output_folder = "history_data"
    os.makedirs(output_folder, exist_ok=True)

    # Save each dictionary as a separate CSV file in the folder
    for key, history in all_histories.items():
        # Convert the dictionary to a pandas DataFrame
        df = pd.DataFrame(history)

        # Format the key for a valid filename
        filename = key.replace('(', '').replace(')', '').replace(', ', '_').replace('_val_', '_validation_') + '.csv'

        # Full path for the CSV file
        file_path = os.path.join(output_folder, filename)

        # Save the DataFrame to a CSV file
        df.to_csv(file_path, index=False)
        print(f"Saved: {file_path}")


    # Save the model
    model.save("traffic_sign_CNN.keras")
    # model.save("traffic_sign_CNN_5Epochs.h5")
    # model.save("traffic_sign_CNN_5Epochs.pb")
    # tf.save_model.save(keras_model)
    return train_model


if __name__ == "__main__":
    main()
