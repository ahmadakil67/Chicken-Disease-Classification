import tensorflow as tf
from pathlib import Path

from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config


    def _valid_generator(self):

        datagenerator_kwargs = dict(
            preprocessing_function=(
                tf.keras.applications.vgg16.preprocess_input
            ),
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=tuple(
                self.config.params_image_size[:-1]
            ),
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
            class_mode="categorical",
            seed=42
        )

        valid_datagenerator = (
            tf.keras.preprocessing.image.ImageDataGenerator(
                **datagenerator_kwargs
            )
        )

        self.valid_generator = (
            valid_datagenerator.flow_from_directory(
                directory=self.config.training_data,
                subset="validation",
                shuffle=False,
                **dataflow_kwargs
            )
        )


    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)


    def evaluation(self):

        self.model = self.load_model(
            self.config.path_of_model
        )

        self._valid_generator()

        self.score = self.model.evaluate(
            self.valid_generator
        )


    def save_score(self):

        scores = {
            "loss": float(self.score[0]),
            "accuracy": float(self.score[1])
        }

        save_json(
            path=Path("scores.json"),
            data=scores
        )