import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, recall_score,f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# 读取数据
data = pd.read_excel('F:/AD_study/participants_data/structure+fun.xlsx')  # 使用您上传的文件路径

# 提取特征和标签
features = data.iloc[:, 6:].values  # 第六列之后的所有脑区的指标数值
labels = data['Group'].values  # 组别信息 AD\MCI\NC
sites = data['center'].values  # 站点信息 S01~S07

# 只保留MCI和NC的数据
mask = (labels == 'AD') | (labels == 'NC')
features = features[mask]
labels = labels[mask]
sites = sites[mask]
feature_names = data.columns[6:].tolist()  # 特征名称列表

# 标签转换为二进制（MCI: 1, NC: 0）
labels = np.where(labels == 'AD', 1, 0)

# 站点列表
unique_sites = np.unique(sites)

# 存储每个站点上的评估指标
results = []

# 定义参数网格
param_grid = {
    'C': [0.1, 1, 10,100],
    'kernel': ['linear', 'rbf','ploy'],
    'gamma': ['scale', 'auto'],
    'degree': [2, 3, 4],  # 仅用于poly核
    'coef0': [0.0, 0.1, 0.5]  # 仅用于poly和sigmoid核
}

# 标准化处理
scaler = StandardScaler()
features = scaler.fit_transform(features)

# 留一站点交叉验证
plt.figure(figsize=(8, 6))

# 存储所有站点的特征重要性
all_importances = []

for site in unique_sites:
    # 划分训练集和测试集
    train_mask = (sites != site)
    test_mask = (sites == site)

    X_train, X_test = features[train_mask], features[test_mask]
    y_train, y_test = labels[train_mask], labels[test_mask]

    # 分层交叉验证
    skf = StratifiedKFold(n_splits=5)

    # 网格搜索
    grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=skf, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # 最佳模型
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # 打印最优参数
    print(f"Site: {site}")
    print(f"Best Parameters: {best_params}")

    # 预测
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)#正确识别负样本
    sensitivity = tp / (tp + fn)#正确识别正样本
    F1_score = f1_score(y_test, y_pred)
    # 计算并绘制ROC曲线
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'Site {site} (AUC = {auc:.2f})')

    print(f"AUC: {auc:.3f}")
    print(f"Accuracy (ACC): {accuracy:.3f}")
    print(f"Sensitivity (SEN): {sensitivity:.3f}")
    print(f"Specificity (SPE): {specificity:.3f}")
    print(f"F1_score: {F1_score:.3f}")

    results.append([site, auc, accuracy, specificity, sensitivity,F1_score])
    # 特征重要性评估
    if best_params['kernel'] == 'linear':
        # 线性核SVM可以直接使用系数作为特征重要性
        importances = np.abs(best_model.coef_[0])
        importance_dict = {name: score for name, score in zip(feature_names, importances)}
    else:
        # 非线性核SVM使用置换重要性
        result = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42)
        importances = result.importances_mean
        importance_dict = {name: score for name, score in zip(feature_names, importances)}

    # 存储当前站点的特征重要性
    all_importances.append(importance_dict)

# 创建DataFrame并保存到CSV文件
columns = ['Site', 'AUC', 'Accuracy', 'Specificity', 'Sensitivity','F1_score']
results_df = pd.DataFrame(results, columns=columns)

#results_df.to_csv('F:/AD_study/SVM_F1/AD_NC/All_AI/S+F/result.csv', index=False)

print(results_df)
print("评估结果已保存到results.csv 文件中。")

# 绘制ROC曲线
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Each Site and Overall')
plt.legend(loc='lower right')
plt.show()

# 特征重要性分析
# 计算所有站点的平均重要性
average_importance = {}
for feature in feature_names:
    scores = [site_importance.get(feature, 0) for site_importance in all_importances]
    average_importance[feature] = np.mean(scores)
# 按重要性排序并选择前30个特征
top_features = sorted(average_importance.items(), key=lambda x: x[1], reverse=True)[:30]
top_feature_names, top_feature_scores = zip(*top_features)
# 绘制特征重要性条形图
plt.figure(figsize=(12, 10))
plt.barh(top_feature_names, top_feature_scores)
plt.xlabel('Feature Importance')
plt.ylabel('Feature Name')
plt.title('Top 30 Features Contributing to the Model')
plt.tight_layout()
#plt.savefig('F:/AD_study/SVM_F1/AD_NC/All_AI/S+F/feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()