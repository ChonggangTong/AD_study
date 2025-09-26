import os
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# 初始化保存所有站点的预测和真实值
all_predictions = []
all_true = []
results = []

# 创建输出目录，确保路径存在
output_dir = 'F:/AD_study/SVR/AD_MCI/p/CT'
os.makedirs(output_dir, exist_ok=True)

for i in range(1, 8):
    print(f"\nProcessing Site S0{i}...")
    # 设置路径，确保路径正确
    base_dir = 'F:/AD_study/Group_difference'
    participants_data_dir = 'F:/AD_study/participants_data'
    indicators = ['CT']

    # Step 1: 提取显著脑区
    significant_regions = {}

    for indicator in indicators:
        file_path = f'{base_dir}/{indicator}/AD_MCI/Leave_One_Out_S0{str(i)}_Results.csv'
        try:
            df = pd.read_csv(file_path)
            # 假设 'p_filter' 为筛选显著性的列，'AI' 为脑区名称
            significant_regions[indicator] = df[df['p_filter'] != 0]['AI'].tolist()
            print(f"Indicator {indicator}: {len(significant_regions[indicator])} significant regions found.")
        except FileNotFoundError:
            print(f"文件未找到: {file_path}")
            significant_regions[indicator] = []
        except Exception as e:
            print(f"读取文件时出错: {file_path}, 错误: {e}")
            significant_regions[indicator] = []

    # Step 2: 构建训练集特征矩阵
    features_train = []
    labels_train = None  # 初始化为 None

    for indicator in indicators:
        # 读取原始AI值文件
        ai_file_path = f'{participants_data_dir}/{indicator}_AI.xlsx'
        try:
            ai_df = pd.read_excel(ai_file_path)
        except FileNotFoundError:
            print(f"AI值文件未找到: {ai_file_path}")
            continue
        except Exception as e:
            print(f"读取AI值文件时出错: {ai_file_path}, 错误: {e}")
            continue

        # 筛选MCI和NC两类数据，且排除当前站点
        train_df = ai_df[(ai_df['Group'].isin(['MCI', 'AD'])) & (ai_df['center'] != f'S0{i}')]

        regions = significant_regions[indicator]

        if not regions:
            print(f"Indicator {indicator}: No significant regions to include.")
            continue

        # 确保所有显著脑区在数据中存在
        existing_regions = [region for region in regions if region in train_df.columns]
        if not existing_regions:
            print(f"Indicator {indicator}: No existing significant regions found in training data.")
            continue

        # 提取显著脑区对应的AI值作为特征
        ai_values_train = train_df[existing_regions].values
        features_train.append(ai_values_train)

        # 提取MMSE分数作为标签
        if labels_train is None:
            labels_train = train_df['MMSE'].values
        else:
            # 确保当前指标的样本顺序与之前一致
            # 这里假设每个指标的样本顺序相同
            if not np.array_equal(labels_train, train_df['MMSE'].values):
                print(f"Indicator {indicator}: Labels do not match with previous indicators.")
                # 根据具体情况处理，如重新对齐样本或跳过该指标
                continue

    if not features_train:
        print(f"站点 S0{i}: 没有可用的训练特征，跳过此站点。")
        continue

    # 将多个结构指标的特征整合为一个完整的训练集特征矩阵
    X_train = np.hstack(features_train)
    y_train = np.array(labels_train)


    print(X_train.shape)
    # Step 3: 构建测试集特征矩阵
    features_test = []
    labels_test = None  # 初始化为 None

    for indicator in indicators:
        # 使用与训练集相同的原始AI值文件
        ai_file_path = f'{participants_data_dir}/{indicator}_AI.xlsx'
        try:
            ai_df = pd.read_excel(ai_file_path)
        except FileNotFoundError:
            print(f"AI值文件未找到: {ai_file_path}")
            continue
        except Exception as e:
            print(f"读取AI值文件时出错: {ai_file_path}, 错误: {e}")
            continue

        # 筛选MCI和NC两类数据，仅保留当前站点的数据
        test_df = ai_df[(ai_df['Group'].isin(['MCI', 'AD'])) & (ai_df['center'] == f'S0{i}')]

        regions = significant_regions[indicator]

        if not regions:
            print(f"Indicator {indicator}: No significant regions to include.")
            continue

        # 确保所有显著脑区在数据中存在
        existing_regions = [region for region in regions if region in test_df.columns]
        if not existing_regions:
            print(f"Indicator {indicator}: No existing significant regions found in test data.")
            continue

        # 提取显著脑区对应的AI值作为特征
        ai_values_test = test_df[existing_regions].values
        features_test.append(ai_values_test)

        # 提取MMSE分数作为标签
        if labels_test is None:
            labels_test = test_df['MMSE'].values
        else:
            # 确保当前指标的样本顺序与之前一致
            if not np.array_equal(labels_test, test_df['MMSE'].values):
                print(f"Indicator {indicator}: Test labels do not match with previous indicators.")
                # 根据具体情况处理，如重新对齐样本或跳过该指标
                continue

    if not features_test:
        print(f"站点 S0{i}: 没有可用的测试特征，跳过此站点。")
        continue

    # 将多个结构指标的特征整合为一个完整的测试集特征矩阵
    X_test = np.hstack(features_test)
    y_test = np.array(labels_test)

    print(X_test.shape)

    # Step 4: 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 定义参数网格
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'ploy'],
        'gamma': ['scale', 'auto'],
        'degree': [2, 3, 4],  # 仅用于poly核
        'coef0': [0.0, 0.1, 0.5]  # 仅用于poly和sigmoid核
    }
    svr = SVR()

    # 使用GridSearchCV寻找最佳参数
    grid_search = GridSearchCV(
        svr,
        param_grid,
        cv=5,  # 可以根据需要调整交叉验证折数
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train_scaled, y_train)

    # 获取最佳模型
    best_svr = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"最佳参数: {best_params}")

    # Step 6: 评估模型性能
    y_pred = best_svr.predict(X_test_scaled)

    # 保存预测结果与真实结果
    all_predictions.extend(y_pred)
    all_true.extend(y_test)

    # 计算评估指标
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    r, p = pearsonr(y_test, y_pred)

    print(f"MAE: {mae:.3f}, MSE: {mse:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}")
    print(f"Pearson Correlation: r = {r:.4f}, p-value = {p:.4e}")

    # 保存每个站点的结果
    results.append(['S0' + str(i), mae, mse, rmse, r2, r, p])

# Step 7: 保存所有站点的预测结果与真实结果
predictions_df = pd.DataFrame({
    'Predicted_MMSE': all_predictions,
    'True_MMSE': all_true
})
predictions_df.to_csv(os.path.join(output_dir, 'all_predictions.csv'), index=False)
print("\n所有预测结果与真实结果已保存至:", os.path.join(output_dir, 'all_predictions.csv'))

# Step 8: 计算整体的Pearson相关系数
overall_r, overall_p = pearsonr(all_true, all_predictions)
print(f"\n整体Pearson相关系数: r = {overall_r:.4f}, p-value = {overall_p:.4e}")

# Step 9: 保存各站点的评估指标
results_df = pd.DataFrame(results, columns=['Site', 'MAE', 'MSE', 'RMSE', 'R²', 'Pearson_r', 'Pearson_p'])
results_df.to_csv(os.path.join(output_dir, 'results.csv'), index=False)
print("各站点评估指标已保存至:", os.path.join(output_dir, 'results.csv'))


