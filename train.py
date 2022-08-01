# Imports
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#-----------Training Setup-----------#
# Prevent CUDA from using GPU as it will crash on the development PC
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Set Constants of the model
BATCH_SIZE = 64
H5_OUTPUT_FILENAME = 'inception_v3_trained.h5'
IMPORT_DATA_FILENAME = 'data_train.npy'
IMPORT_LABELS_FILENAME = 'labels_train.npy'
MAX_EPOCHS = 1000
PATIENCE = 15

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
    labels_names = ['Stop','Yield','Red Light','Green Light','Roundabout','Right Turn Only',
                'Do Not Enter','Crosswalk','Handicap Parking','No Parking']
    return labels_names[np.argmax(np_array)]

# This translates an entire array of one-hot encoded sign predictions
def translate_all(np_array):
    translated_values = []
    for i in np_array:
        translated_values.append(one_hot_translator(i))
    return np.array(translated_values)

def train():
  #-----------Data Preparation-----------#
  # Import the data
  data_train = np.load(IMPORT_DATA_FILENAME).transpose()
  labels_train = np.load(IMPORT_LABELS_FILENAME)

  # Reshape the data into 300x300xRGB images
  data_train = np.array([i.reshape(300,300,3) for i in data_train])

  # Resize the data to fit into a 150x150 area to greatly reduce the calculation requirements
  data_train = np.array(tf.cast(tf.image.resize(data_train,(150,150)), np.uint8))

  # Process the data so that it is in the expected form for the InceptionV3 model
  # This will preprocess all test and training data
  processed = tf.keras.applications.inception_v3.preprocess_input(data_train, data_format=None)

  # Break down data into training and test sets
  x_train, x_test, t_train, t_test = train_test_split(processed, one_hot_training(labels_train), test_size=0.20, random_state=1)


  # Create ImageDataGenerator object to randomly augment the data for training
  # This will add in a horiziontal flip, vertical flip, up to 90deg of rotation and/or a reduction of brightness up to 25%
  train_datagen = ImageDataGenerator(horizontal_flip=True,
                                     vertical_flip=True,
                                     rotation_range=90,
                                     brightness_range=(.75, 1))

  train_generator = train_datagen.flow(
      x_train,
      y = t_train,
      batch_size=BATCH_SIZE)

  # Import the InceptionV3 Model and set the input shape to match the input data
  inception = InceptionV3(input_shape=(150,150,3),
                         include_top=False,
                         weights='imagenet')

  # Set layers to false to prevent overwriting the existing model
  for layer in inception.layers:
      layer.trainable = False

  # Create output layers that will be trained on the data
  x = tf.keras.layers.Flatten()(inception.output)
  x = tf.keras.layers.Dense(1024, activation='relu')(x)
  x = tf.keras.layers.Dropout(0.15)(x)
  x = tf.keras.layers.Dense(10, activation='softmax')(x)

  # Create Optimizer
  optimizer = SGD(learning_rate=0.01, nesterov=True)

  # Finalize and compile the model
  model = tf.keras.Model(inception.input, outputs = x)
  model.compile(optimizer = optimizer,
               loss = 'categorical_crossentropy',
               metrics = ['categorical_accuracy', 'acc','mean_squared_error'])

  # Train the model
  # Model stops training ~113 Epochs
  checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=H5_OUTPUT_FILENAME,monitor='loss',verbose=0,save_best_only=True,mode='min')
  early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=PATIENCE)
  history = model.fit(train_generator, epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stop, checkpoint])
  model.save(H5_OUTPUT_FILENAME)

if __name__ == "__main__":
  train()