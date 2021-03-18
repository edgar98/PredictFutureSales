import data_preparation
import lin_reg_predictor
import lstm_predictor

if __name__ == '__main__':
    x_train, y_train, x_test, test_data = data_preparation.get_data()
    x_train_check = x_test_check = x_train
    y_train_check = y_test_check = y_train
    lstm_predictor.predict(x_train_check, y_train_check, x_test_check, y_test_check, x_test, test_data)
    # x_train_check, y_train_check, x_test_check, y_test_check, x_test, test_data = data_preparation.get_data2()
    # lin_reg_predictor.predict(x_train_check, y_train_check, x_test_check, y_test_check, x_test, test_data)
