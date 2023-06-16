import os
import sys

import dill
import pickle
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging

def save_model(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj=obj,file=file_obj)
        logging.info('Model Saved')    
    except Exception as e:
        raise CustomException(e,sys) # type: ignore
    
def load_model(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)     # type: ignore