# ------------------------------------------------------------------------------
# Name:        Image Segmentation with U-Net
# Purpose:	   Strait forward way to set directories to a unet train and evaluate it all with setting appropriate parametters..
#
# Author:      Morteza Heidari
#
# Created:     11/20/2021
# ------------------------------------------------------------------------------
import keras
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, UpSampling2D, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
from keras.utils import multi_gpu_model
from keras.utils import to_categorical
from keras.models import load_model
from keras.models import model_from_json
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import numpy as np


class unet():
    def __init__(self, input_size, num_classes, num_channels=3, num_gpus=1):
        self.input_size = input_size
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.num_gpus = num_gpus
        self.model = self.get_unet()
        self.model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.summary()
        plot_model(self.model, to_file='model.png')

    def get_unet(self):
        inputs = Input((self.input_size, self.input_size, self.num_channels))
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)
        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
        merge6 = concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv10 = Conv2D(self.num_classes, 1, activation='sigmoid')(conv9)
        model = Model(inputs=inputs, outputs=conv10)
        return model

    def train(self, train_path, val_path, epochs, batch_size, save_path):
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        train_generator = train_datagen.flow_from_directory(
            train_path,
            target_size=(self.input_size, self.input_size),
            batch_size=batch_size,
            class_mode='categorical')
        validation_generator = test_datagen.flow_from_directory(
            val_path,
            target_size=(self.input_size, self.input_size),
            batch_size=batch_size,
            class_mode='categorical')
        self.model.fit_generator(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size,
            callbacks=[TensorBoard(log_dir='logs/{}'.format(self.input_size)),
                       ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True)])

    def predict(self, test_path, save_path):
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        test_generator = test_datagen.flow_from_directory(
            test_path,
            target_size=(self.input_size, self.input_size),
            batch_size=1,
            class_mode='categorical')
        self.model.load_weights(save_path)
        test_generator.reset()
        pred = self.model.predict_generator(test_generator, steps=test_generator.samples)
        return pred

    def save(self, save_path):
        self.model.save_weights(save_path)

    def load(self, save_path):
        self.model.load_weights(save_path)

    def evaluate(self, test_path, save_path):
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        test_generator = test_datagen.flow_from_directory(
            test_path,
            target_size=(self.input_size, self.input_size),
            batch_size=1,
            class_mode='categorical')
        self.model.load_weights(save_path)
        test_generator.reset()
        return self.model.evaluate_generator(test_generator, steps=test_generator.samples)

    def predict_image(self, image_path, save_path):
        img = image.load_img(image_path, target_size=(self.input_size, self.input_size))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        self.model.load_weights(save_path)
        pred = self.model.predict(x)
        return pred


def main():
    model = unet(input_size=256)
    model.train(train_path='data/train', val_path='data/val', epochs=100, batch_size=8, save_path='weights/unet.h5')
    # model.save('weights/unet.h5')
    model.load('weights/unet.h5')
    pred = model.predict(test_path='data/test', save_path='weights/unet.h5')
    print(pred)


def test():
    model = unet(input_size=256)
    model.load('weights/unet.h5')
    pred = model.predict_image('data/test/1.png', 'weights/unet.h5')
    print(pred)
