import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np


def predict(x_train, y_train, x_test, test_data):
    model = LinearRegression().fit(x_train.reshape(-1, x_train.shape[1]), y_train)
    y_train_pred = model.predict(x_train.reshape(-1, x_train.shape[1]))
    print('RMSE valid : %.3f' %
          (np.sqrt(mean_squared_error(y_train, y_train_pred))))
    submission_pfs = model.predict(x_test.reshape(-1, x_test.shape[1]))
    submission_pfs = submission_pfs.clip(0, 20)
    submission = pd.DataFrame({'ID': test_data['ID'], 'item_cnt_month': submission_pfs.ravel()})
    submission.to_csv('sub_pfs_lin_reg.csv', index=False)
