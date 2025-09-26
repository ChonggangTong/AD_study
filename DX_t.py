import pandas as pd

# 读取上传的Excel文件
file_path = 'F:/AD_study/participants_data/FisherZ.xlsx'
data = pd.read_excel(file_path)
data = data[data['Group'].isin(['NC', 'MCI'])]

from sklearn.linear_model import LinearRegression
import numpy as np
from scipy import stats

# 将Group列转换为数字，MCI为0，AD为1
data['DX'] = data['Group'].map({'NC': 0, 'MCI': 1})

# 初始化结果矩阵
result_matrix = np.empty((7, 123)) # 7个站点和105个脑区
result_matrix[:] = np.nan

sites = data['center'].unique()
ai_columns = data.columns[6:-1]
# 对每个站点和每个AI进行线性回归分析
for i, site in enumerate(sites):
    site_data = data[data['center'] == site]
    for j, ai in enumerate(ai_columns):
        X = site_data[['DX', 'Age', 'Sex']]
        y = site_data[ai]
        model = LinearRegression().fit(X, y)
        t_value, _ = stats.ttest_ind(y[X['DX'] == 1], y[X['DX'] == 0], equal_var=False)
        result_matrix[i, j] = t_value

# 创建DataFrame以便于保存
result_df = pd.DataFrame(result_matrix, index=sites, columns=ai_columns)

# 保存结果矩阵到Excel文件
result_file_path = 'F:/AD_study/Group_difference/FZ/MCI_NC/DX_t_values.xlsx'
result_df.to_excel(result_file_path)