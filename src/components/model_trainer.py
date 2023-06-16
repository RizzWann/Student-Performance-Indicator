import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


from src.exception import CustomException
from src.logger import logging
from src.utils import save_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Decision Tree": DecisionTreeRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbour Regression":KNeighborsRegressor() 
            }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter':['best','random'],
                    'max_features':['sqrt','log2'],
                },
                "Linear Regression":{},
                "K-Neighbour Regression":{
                    'n_neighbors':[5,10,15,20],
                    'weights': ['uniform', 'distance'],
                    'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']
                },
           
            }

            model_report:dict=self.evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params
                )
            print(model_report)
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))
            #print(best_model_score)
            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            #print(best_model_name)
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found",sys) # type: ignore
            
            logging.info(f"Best found model on both training and testing dataset")

            save_model(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            #predicted=best_model.predict(X_test)

            #r2_square = r2_score(y_test, predicted)
            
            return (best_model_name,best_model_score)
            
        except Exception as e:
            raise CustomException(e,sys) # type: ignore
           
    def evaluate_models(self,X_train, y_train,X_test,y_test,models:dict,params:dict):
        try:
            report = {}

            for key,model in models.items():
                
                param=params[key]

                gs = GridSearchCV(model,param,cv=3)
                gs.fit(X_train,y_train)

                model.set_params(**gs.best_params_)
                model.fit(X_train,y_train)

                #model.fit(X_train, y_train)  # Train model

                #y_train_pred = model.predict(X_train)

                y_test_pred = model.predict(X_test)

                #train_model_score = r2_score(y_train, y_train_pred)

                test_model_score = r2_score(y_test, y_test_pred)

                report[key] = test_model_score

            return report

        except Exception as e:
            raise CustomException(e, sys)   # type: ignore 