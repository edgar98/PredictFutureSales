import numpy as np
import pandas as pd


def get_data():
    # loading data
    sales_data = pd.read_csv('data/sales_train.csv')
    item_cat = pd.read_csv('data/item_categories.csv')
    items = pd.read_csv('data/items.csv')
    shops = pd.read_csv('data/shops.csv')
    sample_submission = pd.read_csv('data/sample_submission.csv')
    test_data = pd.read_csv('data/test.csv')

    # we can see that 'date' column in sales_data is an object but if we want to manipulate
    # it or want to work on it someway then we have convert it on datetime format
    sales_data['date'] = pd.to_datetime(sales_data['date'], format='%d.%m.%Y')

    # now we will create a pivot tabel by going so we get our data in desired form
    # we want get total count value of an item over the whole month for a shop
    # That why we made shop_id and item_id our indices and date_block_num our column
    # the value we want is item_cnt_day and used sum as aggregating function
    dataset = sales_data.pivot_table(index=['shop_id', 'item_id'], values=['item_cnt_day'], columns=['date_block_num'],
                                     fill_value=0, aggfunc='sum')

    # lets reset our indices, so that data should be in way we can easily manipulate
    dataset.reset_index(inplace=True)

    # lets check on our pivot table
    # dataset.head()

    # Now we will merge our pivot table with the test_data because we want to keep the data of items we have
    # predict
    dataset = pd.merge(test_data, dataset, on=['item_id', 'shop_id'], how='left')

    # lets fill all NaN values with 0
    dataset.fillna(0, inplace=True)
    # lets check our data now
    # dataset.head()

    # we will drop shop_id and item_id because we do not need them
    # we are teaching our model how to generate the next sequence
    dataset.drop(['shop_id', 'item_id', 'ID'], inplace=True, axis=1)
    # dataset.head()

    # X we will keep all columns except the last one
    x_train = np.expand_dims(dataset.values[:, :-1], axis=2)
    # the last column is our label
    y_train = dataset.values[:, -1:]

    # for test we keep all the columns except the first one
    x_test = np.expand_dims(dataset.values[:, 1:], axis=2)

    # lets have a look on the shape
    # print(X_train.shape,y_train.shape,X_test.shape)

    return x_train, y_train, x_test, test_data
