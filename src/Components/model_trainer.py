import os
import sys
from typing import Generator , List ,Tuple 
import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.Constant import *
from src.Exceptions import CustomException
from src.Logger import logging
from src.Utils.main_utils import MainUtils
from src.Utils import read_yaml_file

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

@dataclass
class ModelTrainerConfig:
    artifact_folder=os.path.join(artifact_folder)
    trained_model_path=os.path.join(artifact_folder,"model.pkl")
    expected_accuracy = 0.45
    model_config_file_path=os.path.join('config','model.yaml')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        self.Utils =  MainUtils()
        self.models={
                        'XGBClassifier':XGBClassifier(),
                        'GradientBoostingClassifier':GradientBoostingClassifier(),
                        'SVC':SVC(),
                        'RandomForestClassifier':RandomForestClassifier()
                    }
        

    def evaluate_models(self,X,y,models):
        try:
            logging.info("Evaluation of the models has begun")
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

            logging.info("Collecting the report for each models")
            report={}
            for i in range(len(list(models))):
                model=list(models.values())[i]
                model.fit(X_train,y_train)#train model
                y_test_pred=model.predict(X_test)
                test_model_score=accuracy_score(y_test,y_test_pred)
                report[list(models.keys())[i]]=test_model_score
            logging.info(f"models has been trained and this is their report\n{report}")
            return report
        
        except CustomException as e:
            logging.info('Exception occured in fn:evaluate_models')
            raise CustomException(e,sys)
        

    def get_best_model(self,
                       x_train:np.array,
                       x_test:np.array,
                       y_train:np.array,
                       y_test:np.array):
        
        try:
            model_report : dict = self.evaluate_models(
                x_train=x_train,
                x_test=x_test,
                y_train=y_train,
                y_test=y_test,
                models=self.models
            )

            print(model_report)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model_object = self.models[best_model_name]

            return best_model_name,best_model_object,best_model_score
            
        except CustomException as e:
            logging.info('Exception occured in fn:finetune_best_model')
            raise CustomException(e,sys)
        
    def finetune_best_model(self,best_model_object:object,best_model_name,X_train,y_train)->object:
        try:
            logging.info('Fine tune of the model has begun')
            model_param_grid= read_yaml_file(self.model_trainer_config.model_config_file_path)["model_selection"]["model"][best_model_name]["search_param_grid"]
            grid_search=GridSearchCV(best_model_object,param_grid=model_param_grid,cv=5,n_jobs=-1,verbose=1)
            grid_search.fit(X_train,y_train)
            best_params=grid_search.best_params_
            finetuned_model=best_model_object.set_params(**best_params)
            logging.info(f'Fine tune of the {best_model_name} model has completed')
            logging.info(f"Best parameters for {best_model_name} are as follows\n{best_params}")
            return finetuned_model

        except CustomException as e:
            logging.info('Exception occured in fn:finetune_best_model')
            raise CustomException(e,sys)                   

    

def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('Model training has begun')
            logging.info('Splitting training and testing data into dependent and independent features')
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            logging.info('Evaluating models')
            model_report:dict=self.evaluate_models(X=x_train,y=y_train,models=self.models)
            
            logging.info('Fetching the best model and its score')
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            logging.info(f"best model and its score is\n{best_model_name}:{best_model_score}")
            
            best_model=self.models[best_model_name]

            logging.info('Fine tunning the best model')
            best_model=self.finetune_best_model(
                best_model_name=best_model_name,
                best_model_object=best_model,
                X_train=x_train,
                y_train=y_train
            )
            best_model.fit(x_train,y_train)
            y_pred=best_model.predict(x_test)
            best_model_score=accuracy_score(y_test,y_pred)
            logging.info(f"{best_model_name} and its score:{best_model_score}")

            if best_model_score<0.5:
                raise Exception("No best model found with an accuracy greater than the threshold 0.6")
            
            logging.info("Best model has been found")
            logging.info(f"Saving best model at path:{self.model_trainer_config.trained_model_path}")
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_path),exist_ok=True)
            self.Utils.save_object(file_path=self.model_trainer_config.trained_model_path,obj=best_model)
            logging.info('Best model has been saved sucessfully')
            return self.model_trainer_config.trained_model_path

        except CustomException as e:
            logging.info("Exception occuredin fn:initiate_model_trainer")
            raise CustomException(e,sys)