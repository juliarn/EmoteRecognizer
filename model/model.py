import os

import keras


class KerasModel:

    def __init__(self, model_filepath, image_size, batch_size, training_data):
        self.model_filepath = model_filepath
        self.batch_size = batch_size
        self.training_data = training_data

        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(image_size, image_size, 3)))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Conv2D(64, (3, 3), activation="relu"))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Conv2D(128, (3, 3), activation="relu"))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Conv2D(128, (3, 3), activation="relu"))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(512, activation="relu"))
        model.add(keras.layers.Dense(1, activation="sigmoid"))

        model.compile(loss=keras.losses.binary_crossentropy,
                      optimizer=keras.optimizers.RMSprop(lr=1e-4),
                      metrics=["accuracy"])

        self.model = model

    def create_weights(self):
        if os.path.isfile(self.model_filepath):
            self.model.load_weights(self.model_filepath)
        else:
            train_images, train_labels = self.training_data.get_data()

            train_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,
                                                                           rotation_range=40,
                                                                           width_shift_range=0.2,
                                                                           height_shift_range=0.2,
                                                                           shear_range=0.2,
                                                                           zoom_range=0.2,
                                                                           horizontal_flip=True)
            train_generator = train_generator.flow(train_images, train_labels, self.batch_size)

            self.model.fit_generator(train_generator,
                                     steps_per_epoch=len(train_images) // self.batch_size,
                                     epochs=64,
                                     verbose=1)
            self.model.save_weights(self.model_filepath, True)
