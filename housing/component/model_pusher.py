from housing.logger import logging
from housing.exception import HousingException
from housing.entity.artifact_entity import ModelEvaluationArtifact, ModelPusherArtifact
from housing.entity.config_entity import ModelPusherConfig
import shutil
import os
import sys


class ModelPusher:
    def __init__(self, model_evaluation_artifact: ModelEvaluationArtifact,
                 model_pusher_config: ModelPusherConfig):
        try:
            self.model_evaluation_artifact = model_evaluation_artifact
            self.model_pusher_config = model_pusher_config

        except Exception as e:
            raise HousingException(e, sys) from e

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            evaluated_model_path = self.model_evaluation_artifact.evaluated_model_file_path
            model_file_name = os.path.basename(evaluated_model_path)

            model_export_dir = self.model_pusher_config.export_dir_path
            os.makedirs(model_export_dir, exist_ok=True)
            model_export_file_path = os.path.join(model_export_dir, model_file_name)

            shutil.copy(evaluated_model_path, model_export_file_path)
            logging.info(f"Trained model [{evaluated_model_path}] is copied to [{model_export_file_path}]")

            model_pusher_artifact = ModelPusherArtifact(is_model_pushed=True,
                                                        export_model_file_path=model_export_file_path)
            logging.info(f"Model pusher artifact: [{model_pusher_artifact}]")

            return model_pusher_artifact

        except Exception as e:
            raise HousingException(e, sys) from e

    def __del__(self):
        logging.info("Model pusher log has completed")



