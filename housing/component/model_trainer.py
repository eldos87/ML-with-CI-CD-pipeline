from housing.entity.config_entity import ModelTrainerConfig
from housing.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from housing.logger import logging
from housing.util.util import *
import os
import sys


class HousingEstimatorModel:
    def __init__(self, preprocessing_object, model_object):
        self.preprocessing_object = preprocessing_object
        self.model_object  = model_object

    def predict(self, X):
        """
        This function accepts raw inputs and then transform raw inputs using preprocessing_object
        which guarantee that the inputs are in the same format as the training data.
        Then it perform prediction on transformed features
        """
        transformed_data = self.preprocessing_object.transform(X)
        predictions = self.model_object.predict(transformed_data)
        return predictions

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"  # return object's class name (eg:LinearRegression)

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"


class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        try:
            logging.info(f"Model Training log has started!")
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config

        except Exception as e:
            raise HousingException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            transformed_train_dir = self.data_transformation_artifact.transformed_train_file_path
            transformed_test_dir = self.data_transformation_artifact.transformed_test_file_path
            train_file =

            logging.info(f"Model training Artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise HousingException(e, sys)  from e


