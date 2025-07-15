import pandas as pd
import numpy as np
from Scripts.data_analysis import clean_data, split_data
from Scripts.feature_processing import get_feature
from Scripts.modle import build_model
from Scripts.evaluate import get_regression_model_performance
from sklearn.model_selection import KFold
from Scripts.configuration import conf
from Scripts.utility import dict_to_json, dict_to_table


def run_batch():
    """
    批量训练：对 conf 中每个模型 × 每种编码方式 做 3 折交叉验证，
    结果写入 output.json / output.csv
    """
    # 1. 读数 & 清洗
    data_path = r"C:\Users\ADMIN\Desktop\US-pumpkins.xlsx"
    data = pd.read_excel(data_path, sheet_name="US-pumpkins")
    X, y = clean_data(data)  # 确保这里返回两个值

    outputs = []

    # 2. 遍历配置
    for m_name, m_conf in conf.items():
        print(f"Start training {m_name}")
        for enc_method in m_conf["feature_encoding_methods"]:
            print(f"  feature encoding: {enc_method}")
            ith_output = {
                "model_name": m_name,
                "model_params": m_conf["model_params"],
                "fea_encoding": enc_method
            }

            # 3. 交叉验证
            kf = KFold(n_splits=3, shuffle=True, random_state=0)
            train_rmses, test_rmses = [], []
            train_maes, test_maes = [], []
            train_r2s, test_r2s = [], []

            for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
                train_x, test_x = X.iloc[train_idx], X.iloc[test_idx]
                train_y, test_y = y.iloc[train_idx], y.iloc[test_idx]

                # 4. 特征处理 - 只使用实际存在的列
                available_cat_cols = [col for col in ['City Name', 'Package', 'Variety', 'Color']
                                      if col in train_x.columns]

                train_x_array, test_x_array = get_feature(
                    train_x,
                    test_x,
                    encoding_method=enc_method,
                    encoding_columns=available_cat_cols
                )

                # 5. 建模
                model = build_model(m_name, **m_conf["model_params"])
                model.fit(train_x_array, train_y)

                # 6. 评估
                tr_pred = model.predict(train_x_array)
                te_pred = model.predict(test_x_array)

                tr_rmse, tr_mae, tr_r2 = get_regression_model_performance(train_y, tr_pred)
                te_rmse, te_mae, te_r2 = get_regression_model_performance(test_y, te_pred)

                # 记录单折结果
                ith_output[f"{fold}_fold_train_performance"] = {
                    "rmse": f"{tr_rmse:.2f}",
                    "mae": f"{tr_mae:.2f}",
                    "r2": f"{tr_r2:.2f}"
                }
                ith_output[f"{fold}_fold_test_performance"] = {
                    "rmse": f"{te_rmse:.2f}",
                    "mae": f"{te_mae:.2f}",
                    "r2": f"{te_r2:.2f}"
                }

                train_rmses.append(tr_rmse);
                test_rmses.append(te_rmse)
                train_maes.append(tr_mae);
                test_maes.append(te_mae)
                train_r2s.append(tr_r2);
                test_r2s.append(te_r2)

            # 7. 平均结果
            ith_output["average_train_performance"] = {
                "rmse": f"{np.mean(train_rmses):.2f}",
                "mae": f"{np.mean(train_maes):.2f}",
                "r2": f"{np.mean(train_r2s):.2f}"
            }
            ith_output["average_test_performance"] = {
                "rmse": f"{np.mean(test_rmses):.2f}",
                "mae": f"{np.mean(test_maes):.2f}",
                "r2": f"{np.mean(test_r2s):.2f}"
            }
            outputs.append(ith_output)
            print(f"  ----- {enc_method} done -----")
        print(f"----------- {m_name} training done -------------\n")

    # 8. 保存结果
    dict_to_json(outputs, "./output/output.json")
    dict_to_table(outputs, "./output/output.csv")
    return outputs


def run_single():
    """
    单次快速跑：指定模型 + 编码方式，hold-out 验证
    """
    data_path = r"C:\Users\ADMIN\Desktop\US-pumpkins.xlsx"
    data = pd.read_excel(data_path, sheet_name="US-pumpkins")
    model_name = "RandomForest"
    enc_method = "one-hot"

    X, y = clean_data(data)
    train_x, test_x, train_y, test_y = split_data(X, y)

    # 只使用实际存在的列
    available_cat_cols = [col for col in ['City Name', 'Package', 'Variety', 'Color']
                          if col in train_x.columns]

    train_x_array, test_x_array = get_feature(
        train_x,
        test_x,
        encoding_method=enc_method,
        encoding_columns=available_cat_cols
    )
    print("Train shape:", train_x_array.shape, "Test shape:", test_x_array.shape)

    model = build_model(model_name, **conf[model_name]["model_params"])
    model.fit(train_x_array, train_y)

    print("Train performance")
    tr_pred = model.predict(train_x_array)
    tr_rmse, tr_mae, tr_r2 = get_regression_model_performance(train_y, tr_pred)
    print(f"  RMSE: {tr_rmse:.2f}, MAE: {tr_mae:.2f}, R²: {tr_r2:.2f}")

    print("Test performance")
    te_pred = model.predict(test_x_array)
    te_rmse, te_mae, te_r2 = get_regression_model_performance(test_y, te_pred)
    print(f"  RMSE: {te_rmse:.2f}, MAE: {te_mae:.2f}, R²: {te_r2:.2f}")


if __name__ == "__main__":
    # 二选一即可
    # run_single()
    run_batch()