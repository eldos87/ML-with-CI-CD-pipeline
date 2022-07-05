import logging
from datetime import datetime
import os

log_dir = "Housing_Logs"
os.makedirs(log_dir, exist_ok=True)

now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
file_name = f"log_{now}"
file_path = os.path.join(log_dir, file_name)
file_format = '[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(filename=file_path, filemode='w', format=file_format, level=logging.INFO)


