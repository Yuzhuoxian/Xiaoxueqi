import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import plot_tree
from sklearn.preprocessing import LabelEncoder  # 导入 LabelEncoder


# 数据加载与预处理
def load_and_preprocess_data(data_path):
    # 加载数据
    data = pd.read_excel(data_path, sheet_name="US-pumpkins")

    # 数据清洗与预处理
    def parse_date(date_str):
        try:
            for fmt in ('%m/%d/%y', '%Y-%m-%d', '%m/%d/%Y', '%Y-%m-%d %H:%M:%S'):
                try:
                    return pd.to_datetime(date_str, format=fmt)
                except:
                    continue
            return pd.NaT
        except:
            return pd.NaT

    data['Date'] = data['Date'].apply(parse_date)

    # 转换价格列为数值类型
    price_columns = ['Low Price', 'High Price', 'Mostly Low', 'Mostly High']
    for col in price_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # 提取月份
    data['Month'] = data['Date'].dt.month

    # 清洗类别字段
    data['Item Size'] = data['Item Size'].str.strip().replace('', np.nan)
    data['Color'] = data['Color'].str.strip().replace('', np.nan)

    # 填充缺失值
    data['Color'] = data['Color'].fillna('ORANGE')
    data['Origin'] = data['Origin'].fillna('UNKNOWN')

    # 删除价格缺失或异常的数据
    data = data.dropna(subset=['Low Price', 'High Price'])
    data = data[(data['Low Price'] > 0) & (data['High Price'] < 400)]

    # 新增特征
    data['Avg Price'] = (data['Low Price'] + data['High Price']) / 2
    data['Price Range'] = data['High Price'] - data['Low Price']

    # 特征选择与建模准备
    feature_cols = ['City Name', 'Package', 'Variety', 'Color', 'Month']
    target_col = 'Low Price'

    # 只保留需要的列并删除缺失值
    df_model = data[feature_cols + [target_col]].dropna()

    # 对分类特征进行编码
    label_encoders = {}
    for col in feature_cols:
        if df_model[col].dtype == 'object':
            le = LabelEncoder()
            df_model[col] = le.fit_transform(df_model[col])
            label_encoders[col] = le

    return df_model[feature_cols], df_model[target_col]


# 训练随机森林模型并绘制单棵树的结构图
def train_random_forest_and_plot_tree(X, y):
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练随机森林模型
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 选择第一棵树进行绘制
    tree_index = 0
    plt.figure(figsize=(20, 10))
    plot_tree(model.estimators_[tree_index], filled=True, feature_names=X.columns, max_depth=3)
    plt.title(f"Random Forest Tree {tree_index + 1}")
    plt.show()


# 主函数
def main():
    data_path = r"C:\Users\ADMIN\Desktop\US-pumpkins.xlsx"
    X, y = load_and_preprocess_data(data_path)
    train_random_forest_and_plot_tree(X, y)


if __name__ == "__main__":
    main()