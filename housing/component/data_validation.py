from housing.logger import logging
from housing.exception import HousingException
from housing.entity.config_entity import DataValidationConfig
from housing.entity.artifact_entity import DataValidationArtifact, DataIngestionArtifact
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab
from housing.util.util import read_yaml_file
import os
import sys
import json
import pandas as pd


class DataValidation:
    def __init__(self, data_validation_config: DataValidationConfig,
                 data_ingestion_artifact: DataIngestionArtifact):
        try:
            logging.info(f"Data Validation log has started!")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact

        except Exception as e:
            raise HousingException(e, sys) from e

    def get_train_and_test_df(self):
        try:
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            return train_df, test_df

        except Exception as e:
            raise HousingException(e, sys) from e

    def is_train_test_file_exists(self) -> bool:
        try:
            logging.info("Checking if training and test file is available")

            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            is_train_file_exist = os.path.exists(train_file_path)
            is_test_file_exist = os.path.exists(test_file_path)

            is_available = is_train_file_exist and is_test_file_exist

            logging.info(f"Is train and test file exists?-> {is_available}")

            if not is_available:
                training_file = self.data_ingestion_artifact.train_file_path
                testing_file = self.data_ingestion_artifact.test_file_path
                message = f"Training file: {training_file} or Testing file: {testing_file}" \
                          "is not present"
                raise Exception(message)

            return is_available

        except Exception as e:
            raise HousingException(e, sys) from e

    def validate_number_of_features(self, schema_file) -> None:
        try:
            train_df, test_df = self.get_train_and_test_df()
            number_of_columns_in_schema = len(schema_file['columns'])
            number_of_columns_in_train_file = len(train_df.columns)
            number_of_columns_in_test_file = len(test_df.columns)

            logging.info(f"Number of columns in Schema : [{number_of_columns_in_schema }]")
            logging.info(f"Number of columns in Train dataset : [{number_of_columns_in_train_file}]")
            logging.info(f"Number of columns in Test dataset : [{number_of_columns_in_test_file}]")

            if (number_of_columns_in_schema != number_of_columns_in_train_file) or\
                    (number_of_columns_in_schema != number_of_columns_in_test_file):
                message = f"Mismatch in number of columns between schema file vs train/test file"
                raise Exception(message)

            logging.info(f"Number of columns validated")

        except Exception as e:
            raise HousingException(e, sys) from e

    def validate_feature_names(self, schema_file) -> None:
        try:
            train_df, test_df = self.get_train_and_test_df()
            feature_names_in_schema = list(schema_file['columns'].keys())
            feature_names_in_train_file = train_df.columns
            feature_names_in_test_file = test_df.columns

            for schema_column, train_file_column, test_file_column \
                    in zip(feature_names_in_schema, feature_names_in_train_file, feature_names_in_test_file):
                if (schema_column != train_file_column) or (schema_column != test_file_column):
                    message = f"Mismatch in feature names between schema file vs train/test file"
                    raise Exception(message)
            else:
                logging.info(f"Feature names validated")

        except Exception as e:
            raise HousingException(e, sys) from e

    def validate_feature_types(self, schema_file) -> None:
        try:
            train_df, test_df = self.get_train_and_test_df()
            schema_file_columns = schema_file['columns']
            schema_file_column_keys = schema_file['columns'].keys()
            train_df_dtypes = {k: str(v).replace("dtype(", "").replace(")", "") for k, v in train_df.dtypes.items()}
            test_df_dtypes = {k: str(v).replace("dtype(", "").replace(")", "") for k, v in test_df.dtypes.items()}

            for key in schema_file_column_keys:
                if (schema_file_columns[key] != train_df_dtypes[key]) or\
                        (schema_file_columns[key] != test_df_dtypes[key]):
                    message = f"Mismatch in feature types between schema file vs train/test file"
                    raise Exception(message)
            else:
                logging.info(f"Feature types validated")

        except Exception as e:
            raise HousingException(e, sys) from e

    def validate_values_in_categorical_features(self, schema_file) -> None:
        try:
            train_df, test_df = self.get_train_and_test_df()
            schema_cat_features = schema_file['domain_value']

            for column_name in schema_cat_features.keys():
                if not (set(train_df[column_name].unique()).issubset(set(schema_cat_features[column_name])) or
                        set(test_df[column_name].unique()).issubset(set(schema_cat_features[column_name]))):
                    message = f"Mismatch of categories in categorical features between schema file vs train/test file"
                    raise Exception(message)

            else:
                logging.info(f"Categories in categorical features has validated")

        except Exception as e:
            raise HousingException(e, sys) from e

    def validate_dataset_schema(self) -> bool:
        try:
            schema_path = self.data_validation_config.schema_file_path
            schema_file = read_yaml_file(file_path=schema_path)

            self.validate_number_of_features(schema_file=schema_file)
            self.validate_feature_names(schema_file=schema_file)
            self.validate_feature_types(schema_file=schema_file)
            self.validate_values_in_categorical_features(schema_file=schema_file)

            logging.info(f"Dataset validation against schema file done")
            validation_status = True
            return validation_status

        except Exception as e:
            raise HousingException(e, sys) from e

    def get_and_save_data_drift(self):
        try:
            profile = Profile(sections=[DataDriftProfileSection()])
            train_file, test_file = self.get_train_and_test_df()
            profile.calculate(reference_data=train_file, current_data=test_file)
            report = json.loads(profile.json())   # evidently gives profile.json() as string. Hence load as dict

            report_file_path = self.data_validation_config.report_file_path
            report_dir = os.path.dirname(report_file_path)
            os.makedirs(report_dir, exist_ok=True)

            with open(report_file_path, "w") as report_file:
                json.dump(report, report_file, indent=4)

            return report

        except Exception as e:
            raise HousingException(e, sys) from e

    def save_data_drift_page_report(self):
        try:
            dashboard = Dashboard(tabs=[DataDriftTab()])
            train_df, test_df = self.get_train_and_test_df()
            dashboard.calculate(train_df, test_df)

            report_page_path = self.data_validation_config.report_page_file_path
            report_page_dir = os.path.dirname(report_page_path)
            os.makedirs(report_page_dir, exist_ok=True)
            dashboard.save(filename=report_page_path)

        except Exception as e:
            raise HousingException(e, sys) from e

    def is_data_drift_detected(self):
        try:
            drift_report = self.get_and_save_data_drift()
            self.save_data_drift_page_report()
            data_drift = drift_report['data_drift']['data']['metrics']['dataset_drift']
            logging.info(f"Data drift seen in current dataset? - [{data_drift}]")
            if data_drift:
                raise Exception("Data drift observed in current dataset")

            return data_drift

        except Exception as e:
            raise HousingException(e, sys) from e

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            self.is_train_test_file_exists()
            self.validate_dataset_schema()
            self.is_data_drift_detected()

            data_validation_artifact = DataValidationArtifact(is_validated=True,
                                          message="Data validation successful",
                                          report_file_path=self.data_validation_config.report_file_path,
                                          report_page_file_path=self.data_validation_config.report_page_file_path,
                                          schema_file_path=self.data_validation_config.schema_file_path)

            logging.info(f"Data validation artifact : {data_validation_artifact}")
            return data_validation_artifact

        except Exception as e:
            raise HousingException(e, sys) from e

    def __del__(self):
        logging.info(f"Data validation log completed \n\n")
