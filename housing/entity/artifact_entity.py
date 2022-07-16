from collections import namedtuple

DataIngestionArtifact = namedtuple("DataIngestionArtifact",
                                   ["train_file_path", "test_file_path", "is_ingested", "message"])

DataValidationArtifact = namedtuple("DataValidationArtifact", ["schema_file_path", "report_file_path",
                                                               "report_page_file_path", "is_validated", "message"])

DataTransformationArtifact = namedtuple("DataTransformationArtifact", ["is_transformed", "message",
"transformed_train_file_path", "transformed_test_file_path", "preprocessed_object_file_path"])

ModelTrainerArtifact = namedtuple("ModelTrainerArtifact", ["is_trained", "message", "trained_model_file_path",
                                                           "train_rmse", "test_rmse", "train_accuracy", "test_accuracy"])

ModelEvaluationArtifact = namedtuple("ModelEvaluationArtifact", ["is_model_accepted", "evaluated_model_file_path"])

ModelPusherArtifact = namedtuple("ModelPusherArtifact", ["is_model_pushed", "export_model_file_path"])

MetricInfoArtifact = namedtuple("MetricInfoArtifact", ["model_name", "model_object", "train_rmse", "test_rmse",
                                                       "train_accuracy", "test_accuracy", "index_number"])

InitializedModel = namedtuple("InitializedModel", ["model_number", "model_name", "model_object", "param_grid"])

GridSearchedModel = namedtuple("GridSearchedModel", ["model_number", "model_object", "best_model", "best_params",
                                                     "best_score"])
BestModel = namedtuple("BestModel", ["model_number", "model_object", "best_model", "best_params",
                                                     "best_score"])
