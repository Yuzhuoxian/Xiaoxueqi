# scripts/data_analysis.py
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

def clean_data(df):
    """
    数据清洗与预处理
    """
    # 解析日期
    def parse_date(date_str):
        try:
            for fmt in ('%m/%d/%y', '%Y-%m-%d', '%m/%d/%Y', '%Y-%m-%d %H:%M:%S'):
                try:
                    return datetime.strptime(date_str, fmt)
                except:
                    continue
            return pd.NaT
        except:
            return pd.NaT

    df['Date'] = df['Date'].apply(parse_date)

    # 转换价格列为数值类型
    price_columns = ['Low Price', 'High Price', 'Mostly Low', 'Mostly High']
    for col in price_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 提取月份
    df['Month'] = df['Date'].dt.month

    # 清洗类别字段
    df['Item Size'] = df['Item Size'].str.strip().replace('', np.nan)
    df['Color'] = df['Color'].str.strip().replace('', np.nan)

    # 填充缺失值
    df['Color'] = df['Color'].fillna('ORANGE')
    df['Origin'] = df['Origin'].fillna('UNKNOWN')

    # 删除价格缺失或异常的数据
    df = df.dropna(subset=['Low Price', 'High Price'])
    df = df[(df['Low Price'] > 0) & (df['High Price'] < 400)]

    # 新增特征
    df['Avg Price'] = (df['Low Price'] + df['High Price']) / 2
    df['Price Range'] = df['High Price'] - df['Low Price']

    # 特征选择与建模准备
    feature_cols = ['City Name', 'Package', 'Variety', 'Color', 'Month']
    target_col = 'Low Price'

    # 只保留需要的列并删除缺失值
    df_model = df[feature_cols + [target_col]].dropna()

    # 对分类特征进行编码
    label_encoders = {}
    for col in feature_cols:
        if df_model[col].dtype == 'object':
            le = LabelEncoder()
            df_model[col] = le.fit_transform(df_model[col])
            label_encoders[col] = le

    # 对数值型特征进行填充
    imputer = SimpleImputer(strategy='mean')
    df_model[feature_cols] = imputer.fit_transform(df_model[feature_cols])

    return df_model[feature_cols], df_model[target_col]

def split_data(X, y, test_size=0.2, random_state=42):
    """
    划分训练集和测试集
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)