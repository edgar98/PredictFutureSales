from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import pandas as pd


def lstm_predict(x_train, y_train, x_test, test_data):

    # defining our model
    my_model = Sequential()
    my_model.add(LSTM(units=64, input_shape=(33, 1)))
    my_model.add(Dropout(0.4))
    my_model.add(Dense(1))

    my_model.compile(loss='mse', optimizer='adam', metrics=['mean_squared_error'])

    my_model.fit(x_train, y_train, batch_size=4096, epochs=10)

    # creating submission file
    submission_pfs = my_model.predict(x_test)
    # we will keep every value between 0 and 20
    submission_pfs = submission_pfs.clip(0, 20)
    # creating dataframe with required columns
    submission = pd.DataFrame({'ID': test_data['ID'], 'item_cnt_month': submission_pfs.ravel()})
    # creating csv file from dataframe
    submission.to_csv('sub_pfs.csv', index=False)
