import lstm_predictor
import data_preparation

if __name__ == '__main__':
    x_train, y_train, x_test, test_data = data_preparation.get_data()
    lstm_predictor.lstm_predict(x_train=x_train, y_train=y_train, x_test=x_test, test_data=test_data)