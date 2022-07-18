from flask import Flask
from housing.logger import logging
from housing.exception import HousingException
import sys

app = Flask(__name__)


@app.route('/test', defaults={'req_path': 'housing'})
@app.route('/test/<path:req_path>')
def test(req_path):
    return "Testing new application"


if __name__ == "__main__":
    app.run(debug=True)