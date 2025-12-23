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
logger = setup_logging('Data_Balance')

from imblearn.over_sampling import SMOTE

class BALANCING_DATA:
    @staticmethod
    def balance_data(X_train, y_train):
        try:
            logger.info('Balancing Data')

            # Check if target is categorical
            if y_train.nunique() <= 10:
                smote = SMOTE(random_state=42)
                X_train, y_train = smote.fit_resample(X_train, y_train)
                logger.info('SMOTE applied successfully')
            else:
                logger.info('Target is continuous -> Skipping SMOTE')

            return X_train, y_train

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f'Error in line {error_line.tb_lineno}: {error_msg}')
            return X_train, y_train
