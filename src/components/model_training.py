from pandas import pandas 
from src.exception import CustomException
from src.logger import logging
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import os
import sys
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from dataclasses import dataclass
from src.utils import save_object, evaluate_model
from sklearn.linear_model import Lasso, Ridge

@dataclass
class ModelTrainerConfig:
    train_model_path_config = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("split training and test input data")

            X_train,y_train,X_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = {
                "Linear Regression": LinearRegression(),
                "ridge": Ridge(),
                "Lasso": Lasso(),
                "DecisionTreeRegressor": DecisionTreeRegressor(random_state=42),
                "RandomForestRegressor": RandomForestRegressor(random_state=42),
                "XGBoost": XGBRegressor(random_state=42, n_estimators=100)
          }

            params = {
                "Linear Regression": {},
                "ridge": {
                    "alpha": [0.01, 0.1,1.0,10.0,50.0]
                          },
                "Lasso": {
                    "alpha": [0.01, 0.1,1.0,10.0]
                },
                "DecisionTreeRegressor": {
                    "max_depth": [3, 5, 7, 10],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },
                "RandomForestRegressor": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [3, 5, 7, 10],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },
                "XGBoost": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [3, 5, 7, 10],
                    "learning_rate": [0.01, 0.1, 0.2]
                }
            }

            model_report:dict = evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,params=params)

            best_model_score = max(sorted(list(model_report.values())))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found",sys)

            logging.info("Best found model on training and test dataset")

            save_object(
                file_path = self.model_trainer_config.train_model_path_config,
                obj = best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e,sys)