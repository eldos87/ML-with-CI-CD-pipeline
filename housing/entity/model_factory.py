from housing.exception import HousingException
from housing.logger import logging
from housing.entity.artifact_entity import MetricInfoArtifact, InitializedModel, GridSearchedModel, BestModel
from housing.util.util import read_yaml_file
from housing.constants import *
from sklearn.metrics import r2_score, mean_squared_error
from typing import List
import sys
import importlib
import numpy as np


def evaluate_regression_model(model_list: List, X_train: np.ndarray, y_train: np.ndarray,
                              X_test: np.ndarray, y_test: np.ndarray, base_accuracy: float = 0.6) -> MetricInfoArtifact:
    """
    This function compare multiple models and return best model
    """

    try:
        index_number = 0
        metric_info_artifact = None
        for model in model_list:
            model_name = str(model)    # getting model_name from model object
            logging.info(f"Evaluating model: [{type(model).__name__}]")

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            logging.info(f"{model_name}: train_rmse - [{train_rmse}], test_rmse - [{test_rmse}]")

            train_acc = r2_score(y_train, y_train_pred)
            test_acc = r2_score(y_test, y_test_pred)
            diff_train_test_acc = abs(test_acc - train_acc)

            logging.info(f"{model_name}: train_accuracy - [{train_acc}], test_accuracy - [{test_acc}]")
            logging.info(f"{model_name}: Difference in Train Test accuracy - [{diff_train_test_acc}]")

            if test_acc > base_accuracy and diff_train_test_acc < 0.05:
                base_accuracy = test_acc
                metric_info_artifact = MetricInfoArtifact(model_name=model_name, model_object=model,
                                                          train_rmse=train_rmse, test_rmse=test_rmse,
                                                          train_accuracy=train_acc, test_accuracy=test_acc,
                                                          index_number=index_number)

                logging.info(f"Acceptable model found: {metric_info_artifact}")

        if metric_info_artifact is None:
            logging.info("No acceptable models found")

        return metric_info_artifact

    except Exception as e:
        raise HousingException(e, sys) from e


class ModelFactory:
    def __init__(self, model_config_path: str):
        try:
            self.model_config = read_yaml_file(model_config_path)

            self.gs_module = self.model_config[GRID_SEARCH_KEY][GRID_SEARCH_MODULE_KEY]
            self.gs_class = self.model_config[GRID_SEARCH_KEY][GRID_SEARCH_CLASS_KEY]
            self.gs_params = dict(self.model_config[GRID_SEARCH_KEY][GRID_SEARCH_PARAMS_KEY])

            self.model_list = dict(self.model_config[MODEL_SELECTION_KEY])

        except Exception as e:
            raise HousingException(e, sys) from e

    @staticmethod
    def import_module_class(module_name, class_name):
        try:
            module = importlib.import_module(module_name)
            class_ref = getattr(module, class_name)
            logging.info(f"{class_name} imported from {module}")
            return class_ref

        except Exception as e:
            raise HousingException(e, sys) from e

    def get_initialized_model_list(self) -> List[InitializedModel]:
        """
        This function returns model details listed in model.yaml file
        """
        try:
            initialized_model_list = []
            for model_number in self.model_list.keys():
                model_details = self.model_list[model_number]
                model_name = f"{model_details[MODEL_SELECTION_MODULE_KEY]}.{model_details[MODEL_SELECTION_CLASS_KEY]}"

                model_class_ref = self.import_module_class(model_details[MODEL_SELECTION_MODULE_KEY],
                                                           model_details[MODEL_SELECTION_CLASS_KEY])
                model = model_class_ref()

                param_grid = model_details[MODEL_SELECTION_PARAMS_KEY]

                initialized_model = InitializedModel(model_number=model_number, model_name=model_name,
                                                     model_object=model, param_grid=param_grid)
                initialized_model_list.append(initialized_model)
            return initialized_model_list

        except Exception as e:
            raise HousingException(e, sys) from e

    @staticmethod
    def update_parameters_of_class(instance_ref: object, parameter_data: dict):
        try:
            if not isinstance(parameter_data, dict):
                raise Exception("Please give parameters in dictionary format")
            for key, value in parameter_data.items():
                setattr(instance_ref, key, value)

            return instance_ref

        except Exception as e:
            raise HousingException(e, sys) from e

    def execute_grid_search(self, initialized_model, X, y) -> GridSearchedModel:
        try:
            grid_search_cv_ref = self.import_module_class(module_name=self.gs_module, class_name=self.gs_class)

            grid_search_cv = grid_search_cv_ref(estimator=initialized_model.model_object,
                                                param_grid=initialized_model.param_grid)
            grid_search_cv = self.update_parameters_of_class(instance_ref=grid_search_cv,
                                                             parameter_data=self.gs_params)

            logging.info(f"Starting cross validation on {type(initialized_model.model_object).__name__}")
            grid_search_cv.fit(X, y)
            logging.info(f"Completed cross validation on {type(initialized_model.model_object).__name__}")

            grid_searched_model = GridSearchedModel(model_number=initialized_model.model_number,
                                                    model_object=initialized_model.model_object,
                                                    best_model=grid_search_cv.best_estimator_,
                                                    best_params=grid_search_cv.best_params_,
                                                    best_score=grid_search_cv.best_score_)
            logging.info(f"Grid searched model: {grid_searched_model}")

            return grid_searched_model

        except Exception as e:
            raise HousingException(e, sys) from e

    def perform_hyper_parameter_tuning(self, initialised_model_list: List, X, y) -> List[GridSearchedModel]:
        try:
            grid_searched_model_list = []
            for initialized_model in initialised_model_list:
                grid_searched_model = self.execute_grid_search(initialized_model, X, y)
                grid_searched_model_list.append(grid_searched_model)

            logging.info("Grid Search cross validation completed")
            return grid_searched_model_list

        except Exception as e:
            raise HousingException(e, sys) from e

    # this function has not much importance since best model is chosen based on test set
    @staticmethod
    def get_best_model_on_training_set(grid_searched_model_list: List, base_accuracy) -> BestModel:
        try:
            best_model = None
            for gs_model in grid_searched_model_list:
                if gs_model.best_score > base_accuracy:
                    logging.info(f"Acceptable model found : {gs_model}")
                    base_accuracy = gs_model.best_score
                    best_model = gs_model

            if best_model is None:
                logging.info(f"No acceptable models found")
                raise Exception(f"None of the tested models has base accuracy: {base_accuracy}")

            logging.info(f"Best model : {best_model}")
            return best_model

        except Exception as e:
            raise HousingException(e, sys) from e

    def initiate_best_model_finder(self, X, y, base_accuracy=0.6) -> BestModel:
        try:
            logging.info("Started initialising model from config file")
            initialized_model_list = self.get_initialized_model_list()
            logging.info(f"Initialized model list: {initialized_model_list}")

            grid_searched_model_list = self.perform_hyper_parameter_tuning(initialised_model_list=initialized_model_list,
                                                                           X=X, y=y)

            best_model = self.get_best_model_on_training_set(grid_searched_model_list, base_accuracy)
            return best_model

        except Exception as e:
            raise HousingException(e, sys) from e

