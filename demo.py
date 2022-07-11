from housing.pipeline.pipeline import Pipeline
from housing.config.configuration import Configuration
from housing.exception import HousingException
from housing.logger import logging
import sys

def main():
    try:
        # pipeline = Pipeline()
        # pipeline.run_pipeline()
        data_transform_config = Configuration().get_data_transformation_config()
        print(data_transform_config)

    except Exception as e:
        logging.error(e)
        raise HousingException(e, sys) from e


if __name__ == "__main__":
    main()

