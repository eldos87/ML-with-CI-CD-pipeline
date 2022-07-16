from housing.entity.config_entity import DataTransformationConfig
from housing.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact
from housing.constants import *
from housing.logger import logging
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from housing.util.util import *
import os
import sys
import numpy as np


class FeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True, total_rooms_ix=3, population_ix=5, households_ix=6,
                 total_bedrooms_ix=4, columns=None):
        """
        FeatureGenerator Initialization
        :param add_bedrooms_per_room: bool
        :param total_rooms_ix: int index number of total rooms columns
        :param population_ix: int index number of total population columns
        :param households_ix: int index number of  households columns
        :param total_bedrooms_ix: int index number of bedrooms columns
        """
        try:
            self.columns = columns
            self.add_bedrooms_per_room = add_bedrooms_per_room

            if self.columns is not None:
                self.total_rooms_ix = self.columns.index(COLUMN_TOTAL_ROOMS)
                self.population_ix = self.columns.index(COLUMN_POPULATION)
                self.households_ix = self.columns.index(COLUMN_HOUSEHOLDS)
                self.total_bedrooms_ix = self.columns.index(COLUMN_TOTAL_BEDROOM)

            self.total_rooms_ix = total_rooms_ix
            self.population_ix = population_ix
            self.households_ix = households_ix
            self.total_bedrooms_ix = total_bedrooms_ix

        except Exception as e:
            raise HousingException(e, sys) from e

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        try:
            rooms_per_household = X[:, self.total_rooms_ix] / X[:, self.households_ix]
            population_per_household = X[:, self.population_ix] / X[:, self.households_ix]

            if self.add_bedrooms_per_room:
                bedrooms_per_room = X[:, self.total_bedrooms_ix] / X[:, self.total_rooms_ix]
                generated_feature = np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
                return generated_feature

            else:
                generated_feature = np.c_[X, rooms_per_household, population_per_household]
                return generated_feature

        except Exception as e:
            raise HousingException(e, sys) from e


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            logging.info(f"Data Transformation log started!")
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config

        except Exception as e:
            raise HousingException(e, sys) from e

    def get_data_transformer_object(self) -> ColumnTransformer:
        try:

            schema_path = self.data_validation_artifact.schema_file_path
            schema = read_yaml_file(file_path=schema_path)

            numerical_cols = schema[NUMERICAL_COLUMN_KEY]
            categorical_cols = schema[CATEGORICAL_COLUMN_KEY]

            logging.info(f"Categorical columns: {categorical_cols}")
            logging.info(f"Numerical columns: {numerical_cols}")

            num_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")),
                                           ("feature_generator", FeatureGenerator(
                                               add_bedrooms_per_room=self.data_transformation_config.add_bedroom_per_room,
                                               columns=numerical_cols)),
                                           ("scaler", StandardScaler())])

            cat_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                                           ("encoder", OneHotEncoder()),
                                           ("scaler", StandardScaler(with_mean=False))])

            preprocessing = ColumnTransformer(transformers=([("num_pipeline", num_pipeline, numerical_cols),
                                                             ("cat_pipeline", cat_pipeline, categorical_cols)]))
            logging.info(f"preprocessing object returned")
            return preprocessing

        except Exception as e:
            raise HousingException(e, sys) from e

    def initiate_data_transformation(self):
        try:
            logging.info("obtaining pre-processing object")
            preprocess_obj = self.get_data_transformer_object()

            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            train_df = load_dataframe(file_path=train_file_path)
            test_df = load_dataframe(file_path=test_file_path)
            logging.info("train & test dataframes loaded")

            schema_path = self.data_validation_artifact.schema_file_path
            schema = read_yaml_file(file_path=schema_path)
            logging.info("schema file loaded")

            target = schema[TARGET_COLUMN_KEY]

            logging.info(f"Splitting input and target feature from training and testing dataframe")
            y_train_df = train_df[target]
            X_train_df = train_df.drop(target, axis=1)
            y_test_df = test_df[target]
            X_test_df = test_df.drop(target, axis=1)

            logging.info(f"Applying preprocessing object on train & test features")
            X_train_array = preprocess_obj.fit_transform(X_train_df)
            X_test_array = preprocess_obj.transform(X_test_df)

            logging.info(f"Concatenate pre-processed features array to target array")
            train_array = np.c_[X_train_array, np.array(y_train_df)]
            test_array = np.c_[X_test_array, np.array(y_test_df)]

            transformed_train_dir = self.data_transformation_config.transformed_train_dir
            transformed_test_dir = self.data_transformation_config.transformed_test_dir

            train_file_name = os.path.basename(train_file_path).replace(".csv", ".npz")
            test_file_name = os.path.basename(test_file_path).replace(".csv", ".npz")

            transformed_train_file_path = os.path.join(transformed_train_dir, train_file_name)
            transformed_test_file_path = os.path.join(transformed_test_dir, test_file_name)

            logging.info(f"Saving transformed training and testing array.")
            save_numpy_array_data(file_path=transformed_train_dir, array_data=train_array)
            save_numpy_array_data(file_path=transformed_test_dir, array_data=test_array)

            preprocessed_object_file_path = self.data_transformation_config.preprocessed_object_file_path

            logging.info(f"Saving preprocessed object.")
            save_object(file_path=preprocessed_object_file_path, obj=preprocess_obj)

            data_transformation_artifact = DataTransformationArtifact(is_transformed=True,
                                                                      message="Data Transformation successful",
                                                                      preprocessed_object_file_path=preprocessed_object_file_path,
                                                                      transformed_train_file_path=transformed_train_file_path,
                                                                      transformed_test_file_path=transformed_test_file_path)

            logging.info(f"Data Transformation Artifact : {data_transformation_artifact}")
            return data_transformation_artifact

        except Exception as e:
            raise HousingException(e, sys) from e

    def __del__(self):
        return f"Data Transformation log completed! \n\n"





