from housing.pipeline.pipeline import Pipeline
from housing.exception import HousingException
from housing.config.configuration import Configuration
from housing.logger import logging
import sys


def main():
    try:
        pipeline = Pipeline(config=Configuration())
        pipeline.start()

    except Exception as e:
        logging.error(e)
        raise HousingException(e, sys) from e


if __name__ == "__main__":
    main()
