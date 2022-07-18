from housing.component.data_ingestion import DataIngestion
from housing.component.data_validation import DataValidation
from housing.component.data_transformation import DataTransformation
from housing.component.model_trainer import ModelTrainer
from housing.component.model_evaluation import ModelEvaluation
from housing.component.model_pusher import ModelPusher
from housing.config.configuration import Configuration
from housing.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact,\
    DataTransformationArtifact, ModelTrainerArtifact, ModelEvaluationArtifact, ModelPusherArtifact, Experiment
from housing.exception import HousingException
from housing.logger import logging
from housing.constants import *
from housing.util.util import *

import sys
import uuid
import os
import pandas as pd
from threading import Thread
from datetime import datetime


class Pipeline(Thread):
    experiment: Experiment = Experiment(*([None]*11))
    experiment_file_path = None

    def __init__(self, config: Configuration):
        try:
            artifact_dir = config.training_pipeline_config.artifact_dir
            os.makedirs(artifact_dir, exist_ok=True)
            Pipeline.experiment_file_path = os.path.join(artifact_dir, EXPERIMENT_DIR, EXPERIMENT_FILE_NAME)
            super().__init__(name="pipeline", daemon=False)
            self.config = config

        except Exception as e:
            raise HousingException(e, sys) from e

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            data_ingestion = DataIngestion(data_ingestion_config=self.config.get_data_ingestion_config())

            return data_ingestion.initiate_data_ingestion()

        except Exception as e:
            raise HousingException(e, sys) from e

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            data_validation = DataValidation(data_validation_config=self.config.get_data_validation_config(),
                                             data_ingestion_artifact=data_ingestion_artifact)

            return data_validation.initiate_data_validation()
        except Exception as e:
            raise HousingException(e, sys) from e

    def start_data_transformation(self, data_ingestion_artifact: DataIngestionArtifact,
                                  data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        try:
            data_transformation = DataTransformation(data_ingestion_artifact=data_ingestion_artifact,
                                                 data_validation_artifact=data_validation_artifact,
                                                 data_transformation_config=self.config.get_data_transformation_config())

            return data_transformation.initiate_data_transformation()

        except Exception as e:
            raise HousingException(e, sys) from e

    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            model_trainer = ModelTrainer(data_transformation_artifact=data_transformation_artifact,
                                         model_trainer_config=self.config.get_model_trainer_config())
            return model_trainer.initiate_model_trainer()

        except Exception as e:
            raise HousingException(e, sys) from e

    def start_model_evaluation(self, data_ingestion_artifact: DataIngestionArtifact,
                               data_validation_artifact: DataValidationArtifact,
                               model_trainer_artifact: ModelTrainerArtifact) -> ModelEvaluationArtifact:
        try:
            model_evaluation = ModelEvaluation(model_evaluation_config=self.config.get_model_evaluation_config(),
                                               model_trainer_artifact=model_trainer_artifact,
                                               data_validation_artifact=data_validation_artifact,
                                               data_ingestion_artifact=data_ingestion_artifact)
            return model_evaluation.initiate_model_evaluation()

        except Exception as e:
            raise HousingException(e, sys) from e

    def start_model_pusher(self, model_evaluation_artifact: ModelEvaluationArtifact) -> ModelPusherArtifact:
        try:
            model_pusher = ModelPusher(model_evaluation_artifact=model_evaluation_artifact,
                                       model_pusher_config=self.config.get_model_pusher_config())
            return model_pusher.initiate_model_pusher()

        except Exception as e:
            raise HousingException(e, sys) from e

    def run_pipeline(self):
        try:
            if Pipeline.experiment.running_status:
                logging.info("Pipeline is already running")
                return Pipeline.experiment

            logging.info("Pipeline is starting!")
            Pipeline.experiment = Experiment(experiment_id=str(uuid.uuid4()),
                                             experiment_file_path=Pipeline.experiment_file_path,
                                             running_status=True,
                                             initialization_timestamp=self.config.time_stamp,
                                             artifact_timestamp=self.config.time_stamp,
                                             start_time=datetime.now(),
                                             stop_time=None,
                                             execution_time=None,
                                             model_accuracy=None,
                                             message="Pipeline started",
                                             is_model_accepted=None
                                             )

            logging.info(f"Pipeline Experiment : {Pipeline.experiment}")
            self.save_experiment()

            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_ingestion_artifact=data_ingestion_artifact,
                                                                          data_validation_artifact=data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            model_evaluation_artifact = self.start_model_evaluation(data_ingestion_artifact=data_ingestion_artifact,
                                                                    data_validation_artifact=data_validation_artifact,
                                                                    model_trainer_artifact=model_trainer_artifact)
            if model_evaluation_artifact.is_model_accepted:
                model_pusher_artifact = self.start_model_pusher(model_evaluation_artifact=model_evaluation_artifact)

            logging.info("Pipeline is completed")
            stop_time = datetime.now()
            Pipeline.experiment = Experiment(experiment_id=Pipeline.experiment.experiment_id,
                                             experiment_file_path=Pipeline.experiment_file_path,
                                             running_status=False,
                                             initialization_timestamp=self.config.time_stamp,
                                             artifact_timestamp=self.config.time_stamp,
                                             start_time=Pipeline.experiment.start_time,
                                             stop_time=stop_time,
                                             execution_time=stop_time - Pipeline.experiment.start_time,
                                             model_accuracy=model_trainer_artifact.test_accuracy,
                                             message="Pipeline completed",
                                             is_model_accepted=model_evaluation_artifact.is_model_accepted
                                             )

            logging.info(f"Pipeline Experiment : {Pipeline.experiment}")
            self.save_experiment()

        except Exception as e:
            raise HousingException(e, sys) from e

    def run(self) -> None:
        try:
            self.run_pipeline()

        except Exception as e:
            raise HousingException(e, sys) from e

    @classmethod
    def save_experiment(cls):
        try:
            if Pipeline.experiment.experiment_id is not None:
                experiment_data_dict = Pipeline.experiment._asdict()
                experiment_data_dict = {key: [value] for key, value in experiment_data_dict.items()}
                file_name = os.path.basename(Pipeline.experiment_file_path)
                experiment_data_dict.update({"created_timestamp": [datetime.now()], "file_name": [file_name]})

                experiment_df = pd.DataFrame(experiment_data_dict)

                dir_name = os.path.dirname(Pipeline.experiment_file_path)
                os.makedirs(dir_name, exist_ok=True)

                if os.path.exists(Pipeline.experiment_file_path):
                    experiment_df.to_csv(Pipeline.experiment_file_path, index=False, header=False, mode="a")
                    logging.info("Experiment dataframe has appended")
                else:
                    experiment_df.to_csv(Pipeline.experiment_file_path, index=False, header=True, mode="w")
                    logging.info("Experiment dataframe has created")
            else:
                logging.info("Please start experiment first before proceeding to save experiment")

        except Exception as e:
            raise HousingException(e, sys) from e

    @classmethod
    def get_experiment_status(cls) -> pd.DataFrame:
        try:
            if os.path.exists(Pipeline.experiment_file_path):
                df = pd.read_csv(Pipeline.experiment_file_path)
                return df.tail()
            else:
                return pd.DataFrame()

        except Exception as e:
            raise HousingException(e, sys) from e

