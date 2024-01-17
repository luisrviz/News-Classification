import pandas as pd
from sklearn import preprocessing
from imbalance_data import Sampling


def read_data(data_path):
    df_train = pd.read_json(f'{data_path}News_category_train.json', lines=False)
    df_test = pd.read_json(f'{data_path}News_category_test.json', lines=False)
    df_train = df_train.sample(frac=1)
    # We drop duplicated values
    df_train = df_train[~df_train.duplicated()]

    # We create the new column text and drop the unused columns link and authors
    df_train["text"] = df_train[['headline', 'short_description']].apply(lambda x: ' '.join(x), axis=1)
    df_train = df_train[~df_train["text"].eq(' ')]
    df_train_x = df_train[["text"]]
    df_train_y = df_train[["category"]]
    df_test["text"] = df_test[['headline', 'short_description']].apply(lambda x: ' '.join(x), axis=1)
    df_test_x = df_test[["text"]]
    df_test_y = df_test[["category"]]

    # We handle here the imbalance problem
    sampling = Sampling()
    df_train_x, df_train_y = sampling.undersampling(df_train_x, df_train_y)
    df_train = pd.concat([df_train_x, df_train_y], axis=1)
    df_train = df_train.sample(frac=1)
    df_train_x = df_train[["text"]]
    df_train_y = df_train[["category"]]

    label_encoding = preprocessing.LabelEncoder()
    label_encoding.fit(df_train_y.category)
    df_train_y["category"] = label_encoding.transform(df_train_y["category"])
    df_test_y["category"] = label_encoding.transform(df_test_y["category"])

    return df_train_x, df_train_y, df_test_x, df_test_y, label_encoding
