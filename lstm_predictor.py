from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np


def predict(x_train_check, y_train_check, x_test_check, y_test_check, x_test, test_data):
    x_train_check = x_train_check.values.reshape(x_train_check.shape[0], x_train_check.shape[1], 1)
    x_test_check = x_test_check.values.reshape(x_test_check.shape[0], x_test_check.shape[1], 1)
    x_test = x_test.values.reshape(x_test.shape[0], x_test.shape[1], 1)
    # defining our model
    my_model = Sequential()
    my_model.add(LSTM(units=64, input_shape=(133, 1)))
    my_model.add(Dropout(0.4))
    my_model.add(Dense(1))

    my_model.compile(loss='mse', optimizer='adam', metrics=['mean_squared_error'])

    my_model.fit(x_train_check, y_train_check, batch_size=4096, epochs=10)

    y_train_pred = my_model.predict(x_test_check)
    print('RMSE valid : %.3f' %
          (np.sqrt(mean_squared_error(y_test_check, y_train_pred))))

    # creating submission file
    submission_pfs = my_model.predict(x_test)
    # we will keep every value between 0 and 20
    submission_pfs = submission_pfs.clip(0, 20)
    # creating dataframe with required columns
    submission = pd.DataFrame({'ID': test_data['ID'], 'item_cnt_month': submission_pfs.ravel()})
    # creating csv file from dataframe
    submission.to_csv('sub_pfs_ltsm.csv', index=False)
