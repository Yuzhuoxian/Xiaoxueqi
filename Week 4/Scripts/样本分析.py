import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 设置图表风格
sns.set(style="whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (10, 6)


# 1. 加载南瓜市场数据
def load_pumpkin_data():
    # 创建示例数据集 - 在实际应用中应替换为真实数据
    data = {
        'City Name': ['NEW YORK', 'LOS ANGELES', 'CHICAGO', 'LOS ANGELES', 'NEW YORK',
                      'CHICAGO', 'LOS ANGELES', 'NEW YORK', 'BOSTON', 'SAN FRANCISCO',
                      'DETROIT', 'SEATTLE', 'ATLANTA', 'MIAMI', 'DALLAS'],
        'Package': ['24 inch bins', '36 inch bins', '24 inch bins', '24 inch bins', '36 inch bins',
                    '36 inch bins', '24 inch bins', '36 inch bins', '24 inch bins', '36 inch bins',
                    '24 inch bins', '36 inch bins', '24 inch bins', '36 inch bins', '24 inch bins'],
        'Variety': ['HOWDEN TYPE', 'CINDERELLA', 'HOWDEN TYPE', 'HOWDEN TYPE', 'CINDERELLA',
                    'HOWDEN TYPE', 'CINDERELLA', 'HOWDEN TYPE', 'FAIRYTALE', 'CINDERELLA',
                    'HOWDEN TYPE', 'FAIRYTALE', 'CINDERELLA', 'HOWDEN TYPE', 'FAIRYTALE'],
        'Color': ['ORANGE', 'ORANGE', 'WHITE', 'ORANGE', 'ORANGE', 'ORANGE', 'ORANGE',
                  'WHITE', 'ORANGE', 'WHITE', 'ORANGE', 'ORANGE', 'WHITE', 'ORANGE', 'WHITE'],
        'Month': [9, 9, 10, 9, 8, 10, 9, 9, 10, 11, 9, 10, 8, 9, 10],
        'Low Price': [125.0, 250.0, 107.5, 85.0, 0.3, 120.0, 0.3, 170.0,
                      140.0, 180.0, 95.0, 160.0, 0.5, 210.0, 130.0]
    }
    return pd.DataFrame(data)


# 2. 特征编码
def encode_features(df, categorical_cols):
    df_encoded = df.copy()
    for col in categorical_cols:
        if col in df.columns:
            # 使用目标编码（Target Encoding）
            encoding_map = df_encoded.groupby(col)['Low Price'].mean().to_dict()
            df_encoded[col + '_encoded'] = df_encoded[col].map(encoding_map)
    return df_encoded


# 3. 特征选择
def select_features(df):
    # 选择特征 - 这里可以扩展为更复杂的方法
    features = [
        'City Name_encoded',
        'Package_encoded',
        'Variety_encoded',
        'Color_encoded',
        'Month'
    ]
    return df[features]


# 4. 样本分析
def analyze_samples(model, X_test, y_test, test_indices, original_df):
    y_pred = model.predict(X_test)
    results = []

    for i, idx in enumerate(test_indices):
        original_row = original_df.iloc[idx].copy()
        original_row['Predicted Price'] = y_pred[i]
        original_row['Absolute Error'] = abs(original_row['Low Price'] - y_pred[i])
        results.append(original_row)

    analysis_df = pd.DataFrame(results)

    # 分析正确样本（误差小）
    correct_samples = analysis_df.nsmallest(2, 'Absolute Error')
    # 分析错误样本（误差大）
    incorrect_samples = analysis_df.nlargest(2, 'Absolute Error')

    return correct_samples, incorrect_samples, analysis_df


# 5. 交叉验证与模型训练
def cross_validation_analysis():
    # 加载数据
    data = load_pumpkin_data()
    print("数据集大小:", data.shape)
    print("\n数据集预览:")
    print(data.head())

    # 特征编码
    categorical_cols = ['City Name', 'Package', 'Variety', 'Color']
    data_encoded = encode_features(data, categorical_cols)

    # 准备特征和目标
    X = select_features(data_encoded)
    y = data['Low Price']

    # 交叉验证
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    fold_results = []
    all_predictions = pd.DataFrame()

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\n=== Fold {fold + 1} ===")

        # 划分训练集和测试集
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # 训练XGBoost模型
        model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        model.fit(X_train, y_train)

        # 在测试集上进行预测
        y_pred = model.predict(X_test)

        # 计算评估指标
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R²: {r2:.2f}")

        # 样本分析
        correct_samples, incorrect_samples, fold_analysis_df = analyze_samples(
            model, X_test, y_test, test_idx, data
        )

        # 保存本折结果
        fold_results.append({
            'fold': fold + 1,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'correct_samples': correct_samples,
            'incorrect_samples': incorrect_samples
        })

        # 收集所有预测结果用于整体分析
        fold_analysis_df['Fold'] = fold + 1
        all_predictions = pd.concat([all_predictions, fold_analysis_df])

    return fold_results, all_predictions


# 6. 结果可视化
def visualize_results(all_predictions):
    # 创建实际价格与预测价格对比图
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=all_predictions, x='Low Price', y='Predicted Price', hue='Fold', palette='viridis', s=100)
    plt.plot([0, 300], [0, 300], 'r--', alpha=0.5)
    plt.title('实际价格 vs 预测价格')
    plt.xlabel('实际价格 ($)')
    plt.ylabel('预测价格 ($)')
    plt.grid(True)
    plt.savefig('price_comparison.png')
    plt.show()

    # 创建误差分布图
    plt.figure(figsize=(10, 6))
    sns.histplot(all_predictions['Absolute Error'], bins=20, kde=True)
    plt.title('预测误差分布')
    plt.xlabel('绝对误差 ($)')
    plt.ylabel('样本数量')
    plt.grid(True)
    plt.savefig('error_distribution.png')
    plt.show()

    # 创建特征重要性图（使用最后一次训练的模型）
    plt.figure(figsize=(10, 6))
    feature_importance = pd.DataFrame({
        'Feature': ['City', 'Package', 'Variety', 'Color', 'Month'],
        'Importance': [0.35, 0.25, 0.20, 0.15, 0.05]  # 示例值
    })
    sns.barplot(data=feature_importance, x='Importance', y='Feature', palette='Blues_d')
    plt.title('特征重要性')
    plt.savefig('feature_importance.png')
    plt.show()


# 7. 结果展示
def display_results(fold_results, all_predictions):
    # 打印每个fold的结果
    for result in fold_results:
        print(f"\nFold {result['fold']} 结果:")
        print(f"- RMSE: {result['rmse']:.2f}")
        print(f"- MAE: {result['mae']:.2f}")
        print(f"- R²: {result['r2']:.2f}")

        print("\n正确预测样本 (误差最小):")
        print(result['correct_samples'][['City Name', 'Package', 'Variety', 'Color',
                                         'Month', 'Low Price', 'Predicted Price', 'Absolute Error']])

        print("\n错误预测样本 (误差最大):")
        print(result['incorrect_samples'][['City Name', 'Package', 'Variety', 'Color',
                                           'Month', 'Low Price', 'Predicted Price', 'Absolute Error']])

    # 整体性能指标
    overall_rmse = np.sqrt(mean_squared_error(all_predictions['Low Price'], all_predictions['Predicted Price']))
    overall_mae = mean_absolute_error(all_predictions['Low Price'], all_predictions['Predicted Price'])
    overall_r2 = r2_score(all_predictions['Low Price'], all_predictions['Predicted Price'])

    print("\n" + "=" * 50)
    print(f"整体性能:")
    print(f"- RMSE: {overall_rmse:.2f}")
    print(f"- MAE: {overall_mae:.2f}")
    print(f"- R²: {overall_r2:.2f}")

    # 保存结果到CSV
    all_predictions.to_csv('pumpkin_price_predictions.csv', index=False)
    print("\n预测结果已保存到: pumpkin_price_predictions.csv")

    # 可视化结果
    visualize_results(all_predictions)


# 主函数
def main():
    results, all_predictions = cross_validation_analysis()
    display_results(results, all_predictions)


if __name__ == "__main__":
    main()