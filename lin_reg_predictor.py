import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np


def predict(x_train_check, y_train_check, x_test_check, y_test_check, x_test, test_data):
    model = LinearRegression().fit(x_train_check, y_train_check)
    y_train_pred = model.predict(x_test_check)
    print('RMSE valid : %.3f' %
          (np.sqrt(mean_squared_error(y_test_check, y_train_pred))))
    submission_pfs = model.predict(x_test)
    submission_pfs = submission_pfs.clip(0, 20)
    submission = pd.DataFrame({'ID': test_data['ID'], 'item_cnt_month': submission_pfs.ravel()})
    submission.to_csv('sub_pfs_lin_reg.csv', index=False)
