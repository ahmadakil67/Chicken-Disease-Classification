from pathlib import Path
import tensorflow as tf

from cnnClassifier.entity.config_entity import TrainingConfig


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config


    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )


    def train_valid_generator(self):

        # VGG16 ImageNet preprocessing
        datagenerator_kwargs = dict(
            preprocessing_function=tf.keras.applications.vgg16.preprocess_input,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=tuple(self.config.params_image_size[:-1]),
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
            class_mode="categorical",
            seed=42
        )


        # -----------------------------
        # Validation generator
        # -----------------------------
        valid_datagenerator = (
            tf.keras.preprocessing.image.ImageDataGenerator(
                **datagenerator_kwargs
            )
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )


        # -----------------------------
        # Training generator
        # -----------------------------
        if self.config.params_is_augmentation:

            train_datagenerator = (
                tf.keras.preprocessing.image.ImageDataGenerator(
                    rotation_range=40,
                    horizontal_flip=True,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=0.2,
                    zoom_range=0.2,
                    **datagenerator_kwargs
                )
            )

        else:

            train_datagenerator = (
                tf.keras.preprocessing.image.ImageDataGenerator(
                    **datagenerator_kwargs
                )
            )


        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

        print("Class mapping:", self.train_generator.class_indices)


    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        model.save(path)


    def train(self):

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            validation_data=self.valid_generator
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )