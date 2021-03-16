import data_preparation
import lin_reg_predictor
import lstm_predictor

if __name__ == '__main__':
    x_train, y_train, x_test, test_data = data_preparation.get_data()
    lstm_predictor.predict(x_train=x_train, y_train=y_train, x_test=x_test, test_data=test_data)
    # lin_reg_predictor.predict()
