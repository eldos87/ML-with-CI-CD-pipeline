from housing.logger import logging
from housing.exception import HousingException
from housing.entity.artifact_entity import ModelEvaluationArtifact, ModelTrainerArtifact, DataValidationArtifact,\
    DataIngestionArtifact
from housing.entity.config_entity import ModelEvaluationConfig
from housing.entity.model_factory import evaluate_regression_model
from housing.util.util import *
from housing.constants import *
import os
import sys


class ModelEvaluation:
    def __init__(self, model_evaluation_config: ModelEvaluationConfig, model_trainer_artifact: ModelTrainerArtifact,
                 data_validation_artifact: DataValidationArtifact, data_ingestion_artifact: DataIngestionArtifact):
        try:
            self.model_evaluation_config = model_evaluation_config
            self.model_trainer_artifact = model_trainer_artifact
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact

        except Exception as e:
            raise HousingException(e, sys) from e

    def get_base_model(self):
        try:
            base_model = None
            base_model_file_path = self.model_evaluation_config.model_evaluation_file_path
            if not os.path.exists(base_model_file_path):
                write_yaml_file(file_path=base_model_file_path)
                return base_model

            eval_file_content = read_yaml_file(base_model_file_path)
            eval_file_content = dict() if eval_file_content is None else eval_file_content

            if BEST_MODEL_KEY not in eval_file_content:
                return base_model

            base_model = load_object(eval_file_content[BEST_MODEL_KEY][MODEL_PATH_KEY])
            return base_model

        except Exception as e:
            raise HousingException(e, sys) from e

    def update_evaluation_report(self, model_evaluation_artifact: ModelEvaluationArtifact):
        try:
            base_model_file_path = self.model_evaluation_config.model_evaluation_file_path
            eval_file_content = read_yaml_file(base_model_file_path)
            eval_file_content = dict() if eval_file_content is None else eval_file_content
            logging.info(f"Existing evaluation file content: {eval_file_content}")

            base_model = None
            if BEST_MODEL_KEY in eval_file_content:
                base_model = eval_file_content[BEST_MODEL_KEY]

            eval_result = {BEST_MODEL_KEY: {MODEL_PATH_KEY: model_evaluation_artifact.evaluated_model_file_path}
                           }

            if base_model is not None:
                model_history = {self.model_evaluation_config.time_stamp: base_model}

                if HISTORY_KEY not in eval_file_content:
                    history = {HISTORY_KEY: model_history}
                    eval_result.update(history)
                else:
                    eval_file_content[HISTORY_KEY].update(model_history)

            eval_file_content.update(eval_result)
            logging.info(f"Updated evaluation file: {eval_file_content}")
            write_yaml_file(file_path=base_model_file_path, data=eval_file_content)

        except Exception as e:
            raise HousingException(e, sys) from e

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logging.info(f"Loading train test raw data for evaluation on models")
            train_df = load_dataframe(file_path=self.data_ingestion_artifact.train_file_path)
            test_df = load_dataframe(file_path=self.data_ingestion_artifact.test_file_path)

            schema_path = self.data_validation_artifact.schema_file_path
            schema_file = read_yaml_file(schema_path)
            target_column = schema_file[TARGET_COLUMN_KEY]

            logging.info(f"Splitting raw data as input features and target")
            X_train = train_df.drop(target_column, axis=1)
            y_train = train_df[target_column]
            X_test = test_df.drop(target_column, axis=1)
            y_test = test_df[target_column]

            base_accuracy = self.model_trainer_artifact.test_accuracy  # This is last updated base accuracy

            logging.info(f"Preparing model list for comparison")
            trained_model_path = self.model_trainer_artifact.trained_model_file_path
            trained_model = load_object(trained_model_path)
            base_model = self.get_base_model()
            model_list = [base_model, trained_model]
            logging.info(f"Model list prepared for comparison: {model_list}")

            if base_model is None:
                logging.info(f"Not found any existing models. Hence accepting trained model as base model")
                model_evaluation_artifact = ModelEvaluationArtifact(is_model_accepted=True,
                                                                    evaluated_model_file_path=trained_model_path)
                self.update_evaluation_report(model_evaluation_artifact=model_evaluation_artifact)
                logging.info(f"Model evaluation artifact : {model_evaluation_artifact}")
                return model_evaluation_artifact

            metric_info_artifact = evaluate_regression_model(model_list=model_list, X_train=X_train, y_train=y_train,
                                      X_test=X_test, y_test=y_test, base_accuracy=base_accuracy)

            if metric_info_artifact is None:
                model_evaluation_artifact = ModelEvaluationArtifact(is_model_accepted=False,
                                                                    evaluated_model_file_path=trained_model_path)
                logging.info(f"No acceptable models found. Hence keep base model: {model_evaluation_artifact}")
                return model_evaluation_artifact

            if metric_info_artifact.index_number == 1:
                model_evaluation_artifact = ModelEvaluationArtifact(is_model_accepted=True,
                                                                    evaluated_model_file_path=trained_model_path)
                self.update_evaluation_report(model_evaluation_artifact=model_evaluation_artifact)
                logging.info(f"Trained model updating as new base model: {model_evaluation_artifact}")

            else:
                logging.info(f"Trained model is not better than base model")
                model_evaluation_artifact = ModelEvaluationArtifact(is_model_accepted=False,
                                                                    evaluated_model_file_path=trained_model_path)
            return model_evaluation_artifact

        except Exception as e:
            raise HousingException(e, sys) from e

    def __del__(self):
        logging.info(f"Model evaluation log completed \n\n")

