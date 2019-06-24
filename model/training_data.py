import os
import random

import cv2
import numpy


class TrainingImage:

    def __init__(self, file_path, image_type):
        self.file_path = file_path
        self.image_type = image_type


class TrainingData:

    def __init__(self, peepo_data_dir, kappa_data_dir, image_size):
        peepo_data = [TrainingImage(peepo_data_dir + file_name, 0) for file_name in os.listdir(peepo_data_dir)]
        kappa_data = [TrainingImage(kappa_data_dir + file_name, 1) for file_name in os.listdir(kappa_data_dir)]

        self.image_size = image_size
        self.training_data = peepo_data + kappa_data
        random.shuffle(self.training_data)

    def get_data(self):
        train_images = []
        train_labels = []
        for train_image in self.training_data:
            try:
                image_data = cv2.imread(train_image.file_path, cv2.IMREAD_COLOR)
                train_images.append(cv2.resize(image_data, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC))

                train_labels.append(train_image.image_type)
            except cv2.error:
                print(f"Failed to parse file {train_image.file_path}")

        return numpy.array(train_images), numpy.array(train_labels)
