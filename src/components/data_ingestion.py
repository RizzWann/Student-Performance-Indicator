import os
import sys
import pandas as pd
from src.components.model_trainer import ModelTrainer

from src.exception import CustomException
from src.logger import logging 
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path:str = os.path.join('artifacts','test.csv')
    raw_data_path:str = os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def init_data_ingestion(self):
        logging.info("Entered the data ingestion component")
        try:
            df = pd.read_csv('notebook/data/StudentsPerformance.csv')
            logging.info("Read the dataset from CSV")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
           
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split")
            train_set, test_set = train_test_split(df,test_size=0.2, random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            
            logging.info("Ingestion Completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj = DataIngestion()
    train_path,test_path = obj.init_data_ingestion()

    train_arr, test_arr, _ = DataTransformation().init_data_transformation(train_path=train_path,test_path=test_path)
    modeltrainer=ModelTrainer()
    print(f'Best model:{modeltrainer.initiate_model_trainer(train_arr,test_arr)}')

                
