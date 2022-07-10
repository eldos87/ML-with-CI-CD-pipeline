from housing.entity.config_entity import DataIngestionConfig
from housing.entity.artifact_entity import DataIngestionArtifact
from housing.exception import HousingException
from housing.logger import logging
from six.moves import urllib
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import numpy as np
import os
import sys
import tarfile


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            logging.info("Data Ingestion log started")
            self.data_ingestion_config = data_ingestion_config

        except Exception as e:
            raise HousingException(e, sys) from e

    def download_housing_data(self) -> str:
        try:
            download_url = self.data_ingestion_config.dataset_download_url
            tgz_download_dir = self.data_ingestion_config.tgz_download_dir
            os.makedirs(tgz_download_dir, exist_ok=True)
            housing_file_name = os.path.basename(download_url)
            tgz_file_path = os.path.join(tgz_download_dir, housing_file_name)
            logging.info(f"Downloading file from :[{download_url}] into :[{tgz_file_path}]")
            urllib.request.urlretrieve(url=download_url, filename=tgz_file_path)
            logging.info(f"File :[{tgz_file_path}] has been downloaded successfully.")
            return tgz_file_path

        except Exception as e:
            raise HousingException(e, sys) from e

    def extract_tgz_data(self, tgz_file_path: str) -> None:
        try:
            raw_dir = self.data_ingestion_config.raw_data_dir
            if os.path.exists(raw_dir):
                os.remove(raw_dir)
            os.makedirs(raw_dir)
            logging.info(f"Extracting tgz file: [{tgz_file_path}] into [{raw_dir}]")
            with tarfile.open(tgz_file_path) as tgz_file_obj:
                tgz_file_obj.extractall(raw_dir)
            logging.info(f"tgz file has extracted into [{raw_dir}]")

        except Exception as e:
            raise HousingException(e, sys) from e

    def split_data_as_train_test(self) -> DataIngestionArtifact:
        try:
            raw_file_path = self.data_ingestion_config.raw_data_dir
            file_name = os.listdir(raw_file_path)[0]
            housing_file_path = os.path.join(raw_file_path, file_name)

            logging.info(f"Reading csv file : [{housing_file_path}]")
            df = pd.read_csv(housing_file_path)
            df['cat'] = pd.cut(df['median_income'], bins=[0, 1.5, 3, 4.5, 6, np.inf], labels=[1, 2, 3, 4, 5])

            logging.info(f"Splitting data into train & test split")
            train_set = None
            test_set = None
            splits = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
            for train_indices, test_indices in splits.split(df, df['cat']):
                train_set = df.loc[train_indices].drop('cat', axis=1)
                test_set = df.loc[test_indices].drop('cat', axis=1)

            train_file_path = os.path.join(self.data_ingestion_config.ingested_train_dir, file_name)
            test_file_path = os.path.join(self.data_ingestion_config.ingested_test_dir, file_name)

            if train_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_train_dir, exist_ok=True)
                train_set.to_csv(train_file_path, index=False)
                logging.info(f"Train set exported to [{train_file_path}]")

            if test_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_test_dir, exist_ok=True)
                test_set.to_csv(test_file_path, index=False)
                logging.info(f"Test set exported to [{test_file_path}]")

            data_ingestion_artifact = DataIngestionArtifact(is_ingested="True", message="Ingestion Successful",
                                                            train_file_path=train_file_path,
                                                            test_file_path=test_file_path)
            return data_ingestion_artifact

        except Exception as e:
            raise HousingException(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            tgz_file_path = self.download_housing_data()
            self.extract_tgz_data(tgz_file_path=tgz_file_path)
            return self.split_data_as_train_test()

        except Exception as e:
            raise HousingException(e, sys) from e

    def __del__(self):
        logging.info("Data Ingestion log completed \n\n")
