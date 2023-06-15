import sys
import os
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_model
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preproccesor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self) -> None:
        self.data_trans_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_feature = ['writing score','reading score']
            categorical_feature = [
                'gender',
                'race/ethnicity',
                'parental level of education',
                'lunch',
                'test preparation course'
            ]

            numerical_pipeline = Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
                ],
            )
            categorical_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('onehot',OneHotEncoder()),
                    ('scaller',StandardScaler(with_mean=False))
                ]
            )

            logging.info("Cat and Num pipeline created")

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline',numerical_pipeline,numerical_feature),
                    ('cat_pipeline',categorical_pipeline,categorical_feature)
                ]
            )
            logging.info('Preprocessing Completed')

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)   
        
    def init_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Data read TRain test")

            preprocessor_obj = self.get_data_transformer_object()

            target_column_name = 'math score'
            numerical_feature = ['writing score','reading score']

            X_train = train_df.drop(columns=[target_column_name],axis=1)
            y_train = train_df[target_column_name]

            X_test = test_df.drop(columns=[target_column_name],axis=1)
            y_test = test_df[target_column_name]
           
            preprocessed_X_train = preprocessor_obj.fit_transform(X_train,False)
            preprocessed_X_test  = preprocessor_obj.transform(X_test)

            trained_X  = np.c_[
              preprocessed_X_train,np.array(y_train)
            ]
            tested_X = np.c_[
                preprocessed_X_test,np.array(y_test)
            ] 
            
            logging.info('Preprossed Completed and Object Created')

            save_model(
                file_path = self.data_trans_config.preproccesor_obj_file_path,
                obj = preprocessor_obj
            )
            return(
                trained_X,
                tested_X,
                self.data_trans_config.preproccesor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e,sys) 


