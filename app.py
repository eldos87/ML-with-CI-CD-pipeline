from flask import Flask
from housing.logger import logging
from housing.exception import HousingException
import sys

app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def test():
    try:
        logging.info("Testing started for exception")
        raise Exception("Testing Exception")

    except Exception as e:
        h = HousingException(e, sys)
        logging.info(h.error_msg)
    return "Testing new application"


if __name__ == "__main__":
    app.run(debug=True)