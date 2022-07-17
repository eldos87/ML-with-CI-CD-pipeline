from housing.entity.config_entity import ModelTrainerConfig
from housing.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from housing.entity.model_factory import *
from housing.logger import logging
from housing.exception import HousingException
from housing.util.util import *
import os
import sys


class HousingEstimatorModel:
    def __init__(self, preprocessing_object, model_object):
        self.preprocessing_object = preprocessing_object
        self.model_object = model_object

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
        return f"{type(self.model_object).__name__}()"  # return object's class name (eg:LinearRegression())

    def __str__(self):
        return f"{type(self.model_object).__name__}()"


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
            logging.info(f"Loading transformed training dataset")
            transformed_train_dir = self.data_transformation_artifact.transformed_train_file_path
            train_file = load_numpy_array_data(transformed_train_dir)

            logging.info(f"Loading transformed testing dataset")
            transformed_test_dir = self.data_transformation_artifact.transformed_test_file_path
            test_file = load_numpy_array_data(transformed_test_dir)

            logging.info(f"Splitting input, target feature from training and testing dataset")
            X_train = train_file[:, :-1]
            y_train = train_file[:, -1]

            X_test = test_file[:, :-1]
            y_test = test_file[:, -1]

            logging.info(f"Extracting model config filepath")
            model_config_path = self.model_trainer_config.model_config_file_path

            logging.info(f"Initializing model factory class using : {model_config_path}")
            model_factory = ModelFactory(model_config_path=model_config_path)

            base_accuracy = self.model_trainer_config.base_accuracy
            logging.info(f"Expected minimum accuracy : {base_accuracy}")

            logging.info(f"Initiating model selection")
            best_model_training_set = model_factory.initiate_best_model_finder(X_train, y_train,
                                                                               base_accuracy=base_accuracy)
            logging.info(f"Best model found on training set : {best_model_training_set}")

            logging.info(f"Extracting grid searched model list")
            grid_searched_model_list = model_factory.grid_searched_model_list
            model_list = [model.best_model for model in grid_searched_model_list]
            logging.info(f"Grid searched model list: {model_list}")

            logging.info(f"Evaluating grid searched models on train, test dataset")
            metric_info = evaluate_regression_model(model_list=model_list, X_train=X_train, y_train=y_train,
                                                    X_test=X_test, y_test=y_test, base_accuracy=base_accuracy)
            logging.info(f"Model evaluation completed")

            logging.info(f"Preparing custom object for prediction")
            preprocess_obj = load_object(file_path=self.data_transformation_artifact.preprocessed_object_file_path)
            model_obj = metric_info.model_object
            trained_model_file_path = self.model_trainer_config.trained_model_file_path

            housing_model = HousingEstimatorModel(preprocessing_object=preprocess_obj, model_object=model_obj)
            save_object(file_path=trained_model_file_path, obj=housing_model)
            logging.info(f"Housing model objected saved in : {trained_model_file_path}")

            model_trainer_artifact = ModelTrainerArtifact(is_trained=True, message="Model Training Successful",
                                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                                train_rmse=metric_info.train_rmse, test_rmse=metric_info.test_rmse,
                                train_accuracy=metric_info.train_accuracy, test_accuracy=metric_info.test_accuracy)

            logging.info(f"Model training Artifact: {model_trainer_artifact}")

            return model_trainer_artifact

        except Exception as e:
            raise HousingException(e, sys) from e

    def __del__(self):
        logging.info("Model training log completed! \n\n")



