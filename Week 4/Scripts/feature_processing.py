import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder


def get_feature(train_x, test_x, encoding_method=None, encoding_columns=None):
    # 确保 encoding_columns 是一个列表
    if encoding_columns is None:
        encoding_columns = []

    # 对分类特征进行编码
    if encoding_method == 'ordinal':
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
        oe.fit(train_x[encoding_columns])
        train_x_enc = oe.transform(train_x[encoding_columns])
        test_x_enc = oe.transform(test_x[encoding_columns])
    elif encoding_method == 'one-hot':
        oe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        oe.fit(train_x[encoding_columns])
        train_x_enc = oe.transform(train_x[encoding_columns])
        test_x_enc = oe.transform(test_x[encoding_columns])
    elif encoding_method == 'label':
        le = LabelEncoder()
        # 拟合训练集
        train_x_enc = train_x[encoding_columns].apply(lambda col: le.fit_transform(col.astype(str)))

        # 处理测试集中的未见过的标签
        def transform_with_unknown(col):
            unique_values = le.classes_
            col = col.astype(str)
            # 将未见过的标签映射为一个特定的值（例如 -1）
            col = col.apply(lambda x: x if x in unique_values else 'unknown')
            # 将 'unknown' 映射为一个特定的数值（例如 -1）
            unknown_value = -1
            le.classes_ = np.append(le.classes_, 'unknown')
            transformed = le.transform(col)
            transformed[col == 'unknown'] = unknown_value
            return transformed

        test_x_enc = test_x[encoding_columns].apply(transform_with_unknown)
    else:
        raise ValueError(f"Unsupported encoding method: {encoding_method}")

    # 将编码后的分类特征与非分类特征合并
    non_encoding_columns = [col for col in train_x.columns if col not in encoding_columns]
    train_x_non_enc = train_x[non_encoding_columns].values
    test_x_non_enc = test_x[non_encoding_columns].values

    train_x_array = np.concatenate((train_x_enc, train_x_non_enc), axis=1)
    test_x_array = np.concatenate((test_x_enc, test_x_non_enc), axis=1)

    return train_x_array, test_x_array