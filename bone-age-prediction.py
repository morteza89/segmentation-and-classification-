# ------------------------------------------------------------------------------
# Name:        BONE AGE PREDICTION
# Purpose:	   Predict AGE of the patient with a CNN network in the range of 0 to 18 years old.
# FIne and download the dataset from the following link: https://www.kaggle.com/kmader/rsna-bone-age
# The bone age prediction is based on the bone age dataset from RSNA challenge.
# Author:      Morteza Heidari
#
# Created:     12/2/2021
# ------------------------------------------------------------------------------
# in this function we would like to read training images and their corresponding labels and then train our model
# we would like to define a deep learning model and train it
# we would like to save the model for future use
# we would like to test the model on test data
# we would like to evaluate the model on test data
# we would like to predict the age of a person based on the test data

# import os
from keras.layers.pooling import GlobalMaxPool2D
from keras.losses import mean_absolute_error
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import tensorflow as tf
# from tf.keras.applications.xception import Xception
import pandas as pd
from keras.layers import GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from keras.applications.xception import preprocess_input
from keras.metrics import mean_absolute_error

#####
IMAGE_SIZE = 300
OUTPUT_CLASS = 1  # classification for bone age
print_model_summary = True
plot_model_history = True
BATCH_SIZE = 32
BATCH_SIZE_TEST = 32
EPOCHS = 50
model_name = 'bone_age_model.h5'
OPTIMIZER = 'adam'
plot_predicted_vs_true_age = True
#####


class AgePrediction:
    def __init__(self):
        self.train_data_dir = r'C:\Users\Morteza.Heidari\OneDrive - BioTelemetry, Inc\projects\bone-age\dataset\Bone-Age-Training-Set\boneage-training-dataset'
        self.test_data_dir = r'C:\Users\Morteza.Heidari\OneDrive - BioTelemetry, Inc\projects\bone-age\dataset\Bone-Age-Validation-Set\Bone Age Validation Set\boneage-validation-dataset-1\boneage-validation-dataset-1'
        self.train_labels_path = r'C:\Users\Morteza.Heidari\OneDrive - BioTelemetry, Inc\projects\bone-age\dataset\Bone-Age-Training-Set\train.csv'
        self.test_labels_path = r'C:\Users\Morteza.Heidari\OneDrive - BioTelemetry, Inc\projects\bone-age\dataset\Bone-Age-Validation-Set\Bone Age Validation Set\Validation Dataset.csv'
        self.model_path = './models/model.h5'
        self.results_path = './results/results.csv'
        self.model = None
        self.train_datagen = None
        self.test_datagen = None
        self.train_generator = None
        self.test_generator = None
        self.model_1 = None
        self.model_2 = None

    def read_labels(self):
        self.train_labels = pd.read_csv(self.train_labels_path)
        self.test_labels = pd.read_csv(self.test_labels_path)
        self.train_labels['id'] = self.train_labels['id'].apply(lambda x: str(x) + '.png')
        self.train_labels['gender'] = self.train_labels['male'].apply(lambda x: 'male' if x else 'female')
        self.test_labels['Image ID'] = self.test_labels['Image ID'].apply(lambda x: str(x) + '.png')

    def data_z_score(self):
        mean_bone_age = self.train_labels['boneage'].mean()
        std_bone_age = self.train_labels['boneage'].std()
        self.train_labels['boneage_z'] = (self.train_labels['boneage'] - mean_bone_age) / std_bone_age

    def male_female(self):
        male_train_labels = self.train_labels[self.train_labels['gender'] == 'male']
        female_train_labels = self.train_labels[self.train_labels['gender'] == 'female']
        return male_train_labels, female_train_labels

    def data_split(self):
        self.train_labels['bone_category'] = pd.cut(self.train_labels['boneage'], 10)
        df_train, df_valid = train_test_split(self.train_labels, test_size=0.2, random_state=42)
        return df_train, df_valid

    def generate_data(self):
        df_train, df_valid = self.data_split()
        train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                           rotation_range=30,
                                           width_shift_range=0.1,
                                           height_shift_range=0.1,
                                           shear_range=0.1,
                                           zoom_range=0.1,
                                           horizontal_flip=True,
                                           fill_mode='nearest')
        valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        train_generator = train_datagen.flow_from_dataframe(dataframe=df_train,
                                                            directory=self.train_data_dir,
                                                            x_col='id',
                                                            y_col='boneage_z',
                                                            target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                            batch_size=BATCH_SIZE,
                                                            coolor_mode='rgb',
                                                            class_mode='other')
        valid_generator = valid_datagen.flow_from_dataframe(dataframe=df_valid,
                                                            directory=self.train_data_dir,
                                                            x_col='id',
                                                            y_col='boneage_z',
                                                            target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                            batch_size=BATCH_SIZE,
                                                            color_mode='rgb',
                                                            class_mode='other')
        return train_generator, valid_generator

    def test_data_generator(self):
        test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        test_generator = test_datagen.flow_from_directory(directory=self.test_data_dir,
                                                          target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                          batch_size=BATCH_SIZE_TEST,
                                                          color_mode='rgb',
                                                          class_mode='other')

        return test_generator

    def mae_in_months(self, y_true, y_pred):
        mean_bone_age = self.train_labels['boneage'].mean()
        std_bone_age = self.train_labels['boneage'].std()
        MIM = mean_absolute_error(y_true, y_pred) * std_bone_age + mean_bone_age
        return MIM

    def model_architecture(self):
        self.model_1 = tf.keras.applications.xception.Xception(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                                                               include_top=False,
                                                               weights='imagenet')
        self.model_1.trainable = True
        self.model_2 = Sequential()
        self.model_2.add(self.model_1)
        self.model_2.add(GlobalMaxPool2D())
        self.model_2.add(Flatten())
        self.model_2.add(Dense(256, activation='relu'))
        self.model_2.add(Dropout(0.5))
        self.model_2.add(Dense(32, activation='relu'))
        self.model_2.add(Dense(OUTPUT_CLASS, activation='linear'))
        self.model_2.compile(optimizer=OPTIMIZER, loss='mse', metrics=[self.mae_in_months])
        if print_model_summary:
            self.model_2.summary()

    def train_model(self):
        train_generator, valid_generator = self.generate_data()
        self.model_architecture()
        checkpoint = ModelCheckpoint(self.model_path, monitor='val_accuracy', verbose=1, save_best_only=True,
                                     mode='max')
        history = self.model_2.fit_generator(train_generator,
                                             steps_per_epoch=len(train_generator),
                                             epochs=EPOCHS,
                                             validation_data=valid_generator,
                                             validation_steps=len(valid_generator),
                                             callbacks=[checkpoint])
        if plot_model_history:
            self.plot_model_history(history)

    def plot_model_history(self, history):
        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.title('MAE vs Epoch')
        plt.plot(history.history['loss'], label='Train')
        plt.plot(history.history['val_loss'], label='Test')
        plt.legend()
        plt.show()

    def predict_results(self):
        self.model_2.load_weights(self.model_path)
        df_train, df_valid = self.data_split(self.train_labels)
        mean_bone_age, std_bone_age = self.read_labels()['boneage'].mean(), self.read_labels()['boneage'].std()
        test_x, test_y = next(self.valid_datagen.flow_from_directory(dataframe=df_valid,
                                                                     directory=self.train_data_dir,
                                                                     x_col='id',
                                                                     y_col='boneage_z',
                                                                     target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                                     batch_size=BATCH_SIZE,
                                                                     color_mode='rgb',
                                                                     class_mode='other'))

        pred = (self.model_2.predict(test_x, batch_size=BATCH_SIZE, verbose=True)) * std_bone_age + mean_bone_age
        test_months = mean_bone_age + std_bone_age * np.array(test_y)
        pred_reshape = np.reshape(pred, pred.shape[0])
        print("mean of absolute difference for: ", test_months.shape[0], " cases in test set: ", np.mean(abs(pred_reshape - test_months)))
        ord_ind = np.argsort(test_y)
        if plot_predicted_vs_true_age:
            NUMS_TO_PLOT = 10
            ord_ind = ord_ind[np.linspace(0, ord_ind.shape[0] - 1, num=NUMS_TO_PLOT).astype(int)]
            fig, ax = plt.subplots(np.round(NUMS_TO_PLOT / 2), 2, figsize=(20, 20))
            for (ind, ax_row) in zip(ord_ind, ax.flatten()):
                ax_row.imshow(self.test_x[ind, :, :, 0], cmap='bone')
                ax_row.set_title('Age: %fY\nPredicted Age: %fY' % (test_months[ind] / 12.0,
                                                                   pred[ind] / 12.0))
                ax_row.axis('off')

    def predict_test_set(self):
        self.model_2.load_weights(self.model_path)
        test_generator = self.test_data_generator()
        y_predicted = self.model_2.predict_generator(test_generator)
        predicted = y_predicted.flatten() * self.read_labels()['boneage'].std() + self.read_labels()['boneage'].mean()
        filenames = test_generator.filenames
        results = pd.DataFrame({'id': filenames, 'boneage': predicted})
        results.to_csv(self.results_path, index=False)


def main():
    model = AgePrediction()
    model.read_labels()
    model.data_z_score()
    model.train_model()
    model.predict_results()
    model.predict_test_set()


if __name__ == '__main__':
    main()
