import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
import os
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from log_code import setup_logging
logger = setup_logging('main')

from sklearn.model_selection import train_test_split
from Handling_missing_values import VALUE_MISSING
from var_out import VT_OUT
#from feature_selection import FEATURE_SELECTION
from Data_Balance import BALANCING_DATA
from model_training import common
from sklearn.preprocessing import StandardScaler
import pickle

class MRI:
    def __init__(self, path):
        try:
            self.path = path
            self.df = pd.read_csv(path)
            logger.info("Data loaded")

            logger.info(f'Total Rows in the data : {self.df.shape[0]}')
            logger.info(f'Total columns in the data : {self.df.shape[1]}')
            logger.info(f'Checking Null Values : {self.df.isnull().sum()}')

            #for i in self.df.columns:
                #logger.info(f'{self.df[i].dtype}')

            self.X = self.df.iloc[:, :-1]
            self.y = self.df.iloc[:, -1]

            logger.info(f'X shape : {self.X.shape}')
            logger.info(f'y shape : {self.y.shape}')

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                    random_state=42)

            self.y_train = pd.Series(self.y_train).round().astype(int)
            self.y_test = pd.Series(self.y_test).round().astype(int)

            logger.info(f'{self.X_train.columns}')
            logger.info(f'{self.X_test.columns}')

            logger.info(f'{self.y_train.sample(5)}')
            logger.info(f'{self.y_test.sample(5)}')

            logger.info(f'Training data size : {self.X_train.shape}')
            logger.info(f'Testing data size : {self.X_test.shape}')

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

    def missing_values(self):
        try:
            logger.info(f'Missing Values')
            if self.X_train.isnull().sum().all() > 0 or self.X_test.isnull().sum().all() > 0:
                self.X_train, self.X_test = VALUE_MISSING.random_sample(self.X_train, self.X_test)
            else:
                logger.info(f'No Missing values in X_train and X_test')

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')

    def vt_out(self):
        try:
            logger.info('Variable Transformation Started')
            #for i in self.X_train.columns:
                #logger.info(f'{self.X_train[i].dtype}')

            logger.info(f'Columns of X_train:{self.X_train.columns}')
            logger.info(f'Columns of X_test:{self.X_test.columns}')

            self.X_train, self.X_test = VT_OUT.variable_transform_outliers(self.X_train, self.X_test)

            logger.info(f'===========================================================')
            logger.info(f'Columns of X_train_num:{self.X_train.columns}')
            logger.info(f'Columns of X_test_num:{self.X_test.columns}')
            logger.info('Variable Transformation Completed')

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')

    def data_balance(self):
        try:
            logger.info('Balancing Data Started')
            #logger.info(f"Before SMOTE - Class Distribution: {self.y_train.value_counts().to_dict()}")
            self.X_train,self.y_train=BALANCING_DATA.balance_data(self.X_train,self.y_train)
            #logger.info(f"After SMOTE - Class Distribution: {self.y_train.value_counts().to_dict()}")
            logger.info(f"Balanced X_train shape: {self.X_train.shape}")
            logger.info(f"Balanced y_train shape: {self.y_train.shape}")
            logger.info('Balancing Data Completed')

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')

    def data_scaling(self):
        try:
            logger.info('Data Scaling Started')
            logger.info(f'{self.X_train.shape}')
            logger.info(f'{self.X_test.shape}')
            logger.info(f'Before:{self.X_train}')
            logger.info(f'Before:{self.X_test}')

            # Select numeric columns automatically
            scale_cols = self.X_train.select_dtypes(include=['int64', 'float64']).columns

            logger.info(f'Scaling columns: {scale_cols.tolist()}')

            sc = StandardScaler()
            self.X_train[scale_cols] = sc.fit_transform(self.X_train[scale_cols])
            self.X_test[scale_cols] = sc.transform(self.X_test[scale_cols])

            with open('scaler.pkl', 'wb') as f:
                pickle.dump(sc, f)

            logger.info('Data Scaling Completed')

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')

    def models(self):
        try:
            logger.info(f'Training Started')
            logger.info(f"y_train dtype: {self.y_train.dtype}")
            logger.info(f"y_train unique values: {np.unique(self.y_train)}")
            self.y_train = (self.y_train > 0).astype(int)
            self.y_test = (self.y_test > 0).astype(int)
            common(self.X_train,self.y_train,self.X_test,self.y_test)
            logger.info(f'Training Completed')
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')

if __name__ == "__main__":
    try:
        obj = MRI('C:\\Users\\Rajesh\\Downloads\\Mini Projects\\MRI\\dataset_2.csv')
        obj.missing_values()
        obj.vt_out()
        obj.data_balance()
        obj.data_scaling()
        obj.models()

    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')
