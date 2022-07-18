import os
import sys
import yaml
import pandas as pd
import numpy as np
from housing.exception import HousingException
import dill


def read_yaml_file(file_path: str) -> dict:
    """
    read a yaml file and return the contents as dictionary
    """
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise HousingException(e, sys) from e


def write_yaml_file(file_path: str, data: dict = None) -> None:
    """
    write yaml file using given data in dictionary format
    """
    try:
        dir_name = os.path.dirname(file_path)
        os.makedirs(dir_name, exist_ok=True)
        with open(file_path, "w") as yaml_file:
            if data is not None:
                yaml.dump(data, yaml_file)

    except Exception as e:
        raise HousingException(e, sys) from e


def load_dataframe(file_path: str) -> pd.DataFrame:
    """
    loading pandas dataframe from specified filepath
    """
    try:
        df = pd.read_csv(file_path)
        return df

    except Exception as e:
        raise HousingException(e, sys) from e


def save_numpy_array_data(file_path: str, array_data: np.array) -> None:
    """
    saving numpy array in a given file path
    """
    try:
        dir_name = os.path.dirname(file_path)
        os.makedirs(dir_name, exist_ok=True)
        with open(file_path, "wb") as file:
            np.save(file, array_data)

    except Exception as e:
        raise HousingException(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.array:
    """
    loading numpy array from a given file path
    """
    try:
        with open(file_path, "rb") as file:
            return np.load(file)

    except Exception as e:
        raise HousingException(e, sys) from e


def save_object(file_path: str, obj: object) -> None:
    """
     saving dill object in a given file path
     """
    try:
        dir_name = os.path.dirname(file_path)
        os.makedirs(dir_name, exist_ok=True)
        with open(file_path, "wb") as file:
            dill.dump(obj, file)

    except Exception as e:
        raise HousingException(e, sys) from e


def load_object(file_path: str) -> object:
    """
     loading dill object in a given file path
     """
    try:
        with open(file_path, "rb") as file:
            return dill.load(file)

    except Exception as e:
        raise HousingException(e, sys) from e
