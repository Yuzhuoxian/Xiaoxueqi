# ===========================
# 1. 导入库
# ===========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
import os
import warnings

# 设置环境变量，避免 joblib 警告
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # 设置为你电脑的核心数

# 忽略警告
warnings.filterwarnings("ignore")

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.size'] = 12

# ===========================
# 2. 加载数据
# ===========================
file_path = r"C:\Users\ADMIN\Desktop\US-pumpkins.xlsx"
df = pd.read_excel(file_path, sheet_name="US-pumpkins")

# ===========================
# 3. 数据清洗与预处理
# ===========================
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

# ===========================
# 4. 特征选择与建模准备
# ===========================
feature_cols = ['City Name', 'Package', 'Variety', 'Color', 'Month']
target_col = 'Low Price'

df_model = df[feature_cols + [target_col]].dropna()

# One-Hot 编码
categorical_cols = ['City Name', 'Package', 'Variety', 'Color']
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded = encoder.fit_transform(df_model[categorical_cols])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))
encoded_df.index = df_model.index
df_model = pd.concat([df_model.drop(columns=categorical_cols), encoded_df], axis=1)

# 划分训练集和测试集
X = df_model.drop(columns=[target_col])
y = df_model[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===========================
# 5. 可视化分析
# ===========================

# 5.1 训练集价格分布
plt.figure(figsize=(8, 5))
plt.hist(y_train, bins=30, color='skyblue', edgecolor='black')
plt.title('训练集价格分布')
plt.xlabel('Low Price')
plt.ylabel('频数')
plt.tight_layout()
plt.show()

# 5.2 不同城市的平均价格
plt.figure(figsize=(12, 6))
city_avg = df.groupby('City Name')['Low Price'].mean().sort_values(ascending=False).head(10)
sns.barplot(x=city_avg.values, y=city_avg.index, hue=city_avg.index, palette='viridis', legend=False)
plt.title('不同城市的平均南瓜价格')
plt.xlabel('平均价格（美元）')
plt.ylabel('城市')
plt.tight_layout()
plt.show()

# 5.3 不同包装类型的平均价格
plt.figure(figsize=(12, 6))
package_avg = df.groupby('Package')['Low Price'].mean().sort_values(ascending=False).head(10)
sns.barplot(x=package_avg.values, y=package_avg.index, hue=package_avg.index, palette='magma', legend=False)
plt.title('不同包装类型的平均南瓜价格')
plt.xlabel('平均价格（美元）')
plt.ylabel('包装类型')
plt.tight_layout()
plt.show()

# 5.4 价格随月份变化趋势
plt.figure(figsize=(10, 6))
monthly_price = df.groupby('Month')['Low Price'].mean()
plt.plot(monthly_price.index, monthly_price.values, marker='o', linestyle='-', color='green')
plt.title('南瓜价格随月份变化趋势')
plt.xlabel('月份')
plt.ylabel('平均价格（美元）')
plt.grid(True)
plt.tight_layout()
plt.show()

# 5.5 特征相关性热力图
plt.figure(figsize=(10, 8))
corr = df[['Low Price', 'High Price', 'Avg Price', 'Price Range', 'Month']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('特征相关性热力图')
plt.tight_layout()
plt.show()

# ===========================
# 6. 建模与评估
# ===========================

# 6.1 线性回归
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("【线性回归】")
print("均方误差 (MSE):", round(mse_lr, 2))
print("决定系数 (R²):", round(r2_lr, 2))

# 6.2 随机森林回归
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=2)  # 限制核心数
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("\n【随机森林回归】")
print("均方误差 (MSE):", round(mse_rf, 2))
print("决定系数 (R²):", round(r2_rf, 2))

# 6.3 特征重要性（随机森林）
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1][:10]
plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=X.columns[indices], hue=X.columns[indices], palette='Blues_r', legend=False)
plt.title('随机森林特征重要性')
plt.xlabel('重要性')
plt.ylabel('特征')
plt.tight_layout()
plt.show()

# 6.4 预测值 vs 真实值
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.7, color='orange')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title('随机森林：预测值 vs 真实值')
plt.tight_layout()
plt.show()

# 6.5 残差分析
residuals = y_test - y_pred_rf
plt.figure(figsize=(8, 5))
sns.histplot(residuals, bins=30, kde=True, color='purple')
plt.title('残差分布')
plt.xlabel('残差')
plt.ylabel('频数')
plt.tight_layout()
plt.show()

# ===========================
# 7. KMeans 聚类分析（无监督）
# ===========================
X_cluster = df[['Avg Price', 'Price Range']].dropna()
kmeans = KMeans(n_clusters=3, random_state=42)
df_cluster = df.copy()
df_cluster = df_cluster.loc[X_cluster.index]
df_cluster['Cluster'] = kmeans.fit_predict(X_cluster)

# 可视化聚类结果
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Avg Price', y='Price Range', hue='Cluster', data=df_cluster, palette='Set2')
plt.title('KMeans 聚类结果（按价格特征）')
plt.xlabel('Avg Price')
plt.ylabel('Price Range')
plt.tight_layout()
plt.show()