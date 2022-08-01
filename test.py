# Imports
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.inception_v3 import InceptionV3

#-----------Testing Setup-----------#
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
IMPORT_DATA_FILENAME = 'data_train.npy'
IMPORT_LABELS_FILENAME = 'labels_train.npy'
H5_INPUT_FILENAME = 'inception_v3_trained.h5'
PREDICTED_LABEL_OUTPUT_FILENAME = 'hard_predicted_labels.npy'
THRESHOLD = 0
# THRESHOLD was calculated using training data to determine the minimum single ...
# class confidence level that captures > 90% of correctly graded classes

#-----------Helper Methods-----------#
# Breaks down a list of integer values into a one-hot like format
def one_hot_training(np_array):
    transformed_list = []
    for arr in np_array:
        new_arr = np.zeros(10)
        new_arr[int(arr)] = 1
        transformed_list.append(new_arr)
    return np.array(transformed_list)

# This translates the highest value from the one-hot encoding into the correct sign name
def one_hot_translator(np_array):
    max_index = np.argmax(np_array)
    max_value = np.max(np_array)
    if max_value < THRESHOLD:
        return -1
    return max_index

# This translates an entire array of one-hot encoded sign predictions
def translate_all(np_array):
    translated_values = []
    for i in np_array:
        translated_values.append(one_hot_translator(i))
    return np.array(translated_values)

def calculate_acc(true_labels, predicted_labels):
    accurate_labels = 0
    total_labels = len(predicted_labels)
    for index, label in enumerate(predicted_labels):
        if(true_labels[index] == predicted_labels[index]):
            accurate_labels = accurate_labels + 1
    return accurate_labels / total_labels

def test():
    #-----------Data Preparation-----------#
    # Import the data
    data_test = np.load(IMPORT_DATA_FILENAME)
    labels_test = np.load(IMPORT_LABELS_FILENAME)

    # Determine if the test data is the correct shape
    if not(data_test.shape[0] == labels_test.shape[0]):
        data_test = data_test.transpose()

    # Reshape the data into 300x300xRGB images
    data_test = np.array([i.reshape(300,300,3) for i in data_test])

    # Resize the data to fit into a 150x150 area to greatly reduce the calculation requirements
    data_test = np.array(tf.cast(tf.image.resize(data_test,(150,150)), np.uint8))

    # Process the data so that it is in the expected form for the InceptionV3 model
    # This will preprocess all test and training data
    processed = tf.keras.applications.inception_v3.preprocess_input(data_test, data_format=None)

    # Load the pretrained model
    model = load_model(H5_INPUT_FILENAME)

    # Predict the labels given the test data
    predictions = model.predict(processed)

    # Convert the one hot encoded results into the final classifiers
    translated_predictions = translate_all(predictions)

    # Calculate the accuracy by comparing the true and predicted labels
    evaluation = calculate_acc(labels_test, translated_predictions)
    print("\nTest run accuracy is {:.2f}%\n".format(100*evaluation))

    # Print the predicted labels to the screen
    np.set_printoptions(threshold=np.inf)
    print("Predicted labels")
    print(translated_predictions)

    # Save the predicted labels to a new file
    print("\nPredicted labels saved to {}".format(PREDICTED_LABEL_OUTPUT_FILENAME))
    np.save(PREDICTED_LABEL_OUTPUT_FILENAME,translated_predictions)

if __name__ == "__main__":
    test()