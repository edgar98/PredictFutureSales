import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


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


def get_data2():
    train = pd.read_csv(r'./data/sales_train.csv')
    test = pd.read_csv(r'./data/test.csv')
    sample_sub = pd.read_csv(r'./data/sample_submission.csv')
    items = pd.read_csv(r'./data/items.csv')
    items_cat = pd.read_csv(r'./data/item_categories.csv')
    shops = pd.read_csv(r'./data/shops.csv')

    test_shops = test.shop_id.unique()  # в train выборке у нас есть магазины и товары которых нет в test выборке
    train = train[train.shop_id.isin(test_shops)]  # поэтому мы их удалим
    test_items = test.item_id.unique()
    train = train[train.item_id.isin(test_items)]
    train.loc[train.shop_id == 0, 'shop_id'] = 57
    test.loc[test.shop_id == 0, 'shop_id'] = 57
    # Якутск ТЦ "Центральный"
    train.loc[train.shop_id == 1, 'shop_id'] = 58
    test.loc[test.shop_id == 1, 'shop_id'] = 58
    # Жуковский ул. Чкалова 39м²
    train.loc[train.shop_id == 10, 'shop_id'] = 11
    test.loc[test.shop_id == 10, 'shop_id'] = 11

    def split_city(str):
        return str.split(sep=" ", maxsplit=1)[0]

    def split_shop(str):
        return str.split(sep=" ", maxsplit=1)[1]

    def split_item_cat1(str):
        return str.split(sep="-", maxsplit=1)[0]

    def split_item_cat2(str):
        splitted = str.split(sep="-", maxsplit=1)
        if len(splitted) == 1:
            return "No info"
        else:
            return splitted[1]

    def prepare_data(data):  # функция для объединения таблиц и создания новых признаков из старых
        full_items = items.merge(items_cat, left_on="item_category_id", right_on="item_category_id")
        full_data = data.merge(shops, left_on="shop_id", right_on="shop_id").merge(full_items, left_on="item_id",
                                                                                   right_on="item_id")
        del full_items
        full_data['city'] = full_data['shop_name'].apply(split_city)
        full_data['new_shop_name'] = full_data['shop_name'].apply(split_shop)
        full_data['item_cat1'] = full_data['item_category_name'].apply(split_item_cat1)
        full_data['item_cat2'] = full_data['item_category_name'].apply(split_item_cat2)
        full_data.drop(['shop_id', 'item_id', 'shop_name', 'item_name', 'item_category_id', 'item_category_name'],
                       axis=1, inplace=True)
        return full_data

    new_train = prepare_data(train.copy())
    new_test = prepare_data(test.copy())

    new_train = new_train[new_train.item_price < 100000]
    new_train = new_train[new_train.item_cnt_day < 1001]
    ###

    new_test['date_block_num'] = 34  # добавляем порядковый номер месяца в test
    new_test.drop(['ID'], axis=1, inplace=True)
    new_train.drop(['date'], axis=1, inplace=True)
    new_train['item_cnt_day'] = new_train['item_cnt_day']\
        .clip(0, 20)  # преобразуем значения item_cnt_day в необходимый формат > 0
    new_train['month'] = new_train['date_block_num'] % 12  # добавляем номер месяца в train
    new_test['month'] = new_test['date_block_num'] % 12  # добавляем номер месяца в test
    new_train.drop(['item_price'], axis=1, inplace=True)

    X_train = new_train.drop(['item_cnt_day'], axis=1)  # разделение на X и Y
    Y_train = new_train['item_cnt_day']
    X_test = new_test

    cat_features = ['city', 'new_shop_name', 'item_cat1', 'item_cat2']

    def into_numbers(data):  # приводим к необходимому формату категориальные признаки
        num_data = pd.concat([data, pd.get_dummies(data['city'])], axis=1)
        num_data = pd.concat([num_data, pd.get_dummies(data['item_cat1'])], axis=1)
        num_data = pd.concat([num_data, pd.get_dummies(data['item_cat2'])], axis=1)
        num_data = pd.concat([num_data, pd.get_dummies(data['new_shop_name'])], axis=1)
        num_data.drop(cat_features, axis=1, inplace=True)
        return num_data

    X_train_num = into_numbers(X_train.copy())
    X_test_num = into_numbers(X_test.copy())

    X_train_num[' Гарнитуры/Наушники'] = 0
    X_train_num['PC '] = 0
    X_train_num['Игры MAC '] = 0

    X_train_num = X_train_num.reindex(sorted(X_train_num.columns), axis=1)
    X_test_num = X_test_num.reindex(sorted(X_test_num.columns), axis=1)

    def transliterate(name):
        # Словарь с заменами
        dictionary = {'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'e',
                      'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'i', 'к': 'k', 'л': 'l', 'м': 'm', 'н': 'n',
                      'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u', 'ф': 'f', 'х': 'h',
                      'ц': 'c', 'ч': 'cz', 'ш': 'sh', 'щ': 'scz', 'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e',
                      'ю': 'u', 'я': 'ja', 'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D', 'Е': 'E', 'Ё': 'E',
                      'Ж': 'ZH', 'З': 'Z', 'И': 'I', 'Й': 'I', 'К': 'K', 'Л': 'L', 'М': 'M', 'Н': 'N',
                      'О': 'O', 'П': 'P', 'Р': 'R', 'С': 'S', 'Т': 'T', 'У': 'U', 'Ф': 'F', 'Х': 'H',
                      'Ц': 'C', 'Ч': 'CZ', 'Ш': 'SH', 'Щ': 'SCH', 'Ъ': '', 'Ы': 'y', 'Ь': '', 'Э': 'E',
                      'Ю': 'U', 'Я': 'YA', ',': '', '?': '', ' ': '_', '~': '', '!': '', '@': '', '#': '',
                      '$': '', '%': '', '^': '', '&': '', '*': '', '(': '', ')': '', '-': '', '=': '', '+': '',
                      ':': '', ';': '', '<': '', '>': '', '\'': '', '"': '', '\\': '', '/': '', '№': '',
                      '[': '', ']': '', '{': '', '}': '', 'ґ': '', 'ї': '', 'є': '', 'Ґ': 'g', 'Ї': 'i',
                      'Є': 'e', '—': ''}

        # Циклически заменяем все буквы в строке
        for key in dictionary:
            name = name.replace(key, dictionary[key])
        return name

    eng_cols = {}
    for i in X_train_num.columns:
        eng_cols[str(i)] = transliterate(i)

    X_train_num.rename(columns=eng_cols, inplace=True)
    X_test_num.rename(columns=eng_cols, inplace=True)

    X_train_check, X_test_check, y_train_check, y_test_check = train_test_split(X_train_num, Y_train,
                                                                                test_size=0.33,
                                                                                random_state=42)  # нормализованные данные с использованием метода обработки категориальных признаков - get_dummies

    return X_train_check, y_train_check, X_test_check, y_test_check, X_test_num, test
