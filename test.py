# Imports
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import load_model

#-----------Testing Setup-----------#
IMPORT_DATA_FILENAME = 'data_train.npy'
IMPORT_LABELS_FILENAME = 'labels_train.npy'
H5_INPUT_FILENAME = 'inception_v3_trained.h5'
PREDICTED_LABEL_OUTPUT_FILENAME = 'predicted_labels.npy'
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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
    return np.argmax(np_array)

# This translates the highest value from the one-hot encoding into the correct sign name
def one_hot_translator_signs(np_array):
    labels_names = ['Stop','Yield','Red Light','Green Light','Roundabout','Right Turn Only',
                'Do Not Enter','Crosswalk','Handicap Parking','No Parking']
    return labels_names[np.argmax(np_array)]

# This translates an entire array of one-hot encoded sign predictions
def translate_all(np_array):
    translated_values = []
    for i in np_array:
        translated_values.append(one_hot_translator(i))
    return np.array(translated_values)


#-----------Data Preparation-----------#
# Import the data
data_test = np.load(IMPORT_DATA_FILENAME)
labels_test = np.load(IMPORT_LABELS_FILENAME)

if not(data_test.shape[0] == labels_test.shape[0]):
    data_test = data_test.transpose()

# Reshape the data into 300x300xRGB images
data_test = np.array([i.reshape(300,300,3) for i in data_test])

# Resize the data to fit into a 150x150 area to greatly reduce the calculation requirements
data_test = np.array(tf.cast(tf.image.resize(data_test,(150,150)), np.uint8))

# Break down data into training and test sets
x_train, x_test, t_train, t_test = train_test_split(data_test, one_hot_training(labels_test), test_size=0.20, random_state=1)

# Process the data so that it is in the expected form for the InceptionV3 model
# This will preprocess all test and training data
processed = tf.keras.applications.inception_v3.preprocess_input(x_test, data_format=None)

model = load_model(H5_INPUT_FILENAME)
predictions = model.predict(processed)
print(predictions.shape)
translated_predictions = translate_all(predictions)
evaluation = model.evaluate(processed, t_test)
print("\nTest run accuracy is {:.2f}%\n".format(100*evaluation[-2]))
np.set_printoptions(threshold=np.inf)
print("Predicted labels")
print(translated_predictions)
print("\nPredicted labels saved to {}".format(PREDICTED_LABEL_OUTPUT_FILENAME))
np.save(PREDICTED_LABEL_OUTPUT_FILENAME,translated_predictions)

import matplotlib.pyplot as plt
plt.hist(translated_predictions)