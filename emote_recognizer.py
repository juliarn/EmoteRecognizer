import enum

import cv2
import numpy
import requests

import config
from model.model import KerasModel
from model.training_data import TrainingData


class EmoteType(enum.Enum):
    PEEPO = 0
    KAPPA = 1


class EmoteRecognizer:

    def __init__(self):
        training_data = TrainingData(config.peepo_data_dir, config.kappa_data_dir, config.image_size)

        keras_model = KerasModel(config.model_filepath, config.image_size, config.batch_size, training_data)
        keras_model.create_weights()

        self.model = keras_model.model
        self.image_size = config.image_size

    def recognize(self, image_array):
        prediction = self.model.predict(image_array)[0][0]
        return EmoteType(prediction)

    def parse_image(self, url):
        response = requests.get(url)
        image = numpy.asarray(bytearray(response.content), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return numpy.expand_dims(cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC), 0)
