import pandas as pd
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import MinMaxScaler

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

def plot_joint_results(df):
    df[features] = df[features].fillna(df[features].mean())

    df[features] = df[features].clip(lower=0)

    data_tensor = torch.tensor(df[features].values, dtype=torch.float32)

    mask = (data_tensor >= 0).all(dim=1)
    data_tensor = data_tensor[mask]

    for col in range(data_tensor.shape[1]):
        q1 = data_tensor[:, col].quantile(0.25)
        q3 = data_tensor[:, col].quantile(0.75)
        iqr = q3 - q1
        col_mask = (data_tensor[:, col] >= q1 - 1.5 * iqr) & (data_tensor[:, col] <= q3 + 1.5 * iqr)
        data_tensor = data_tensor[col_mask]

    clean_data = data_tensor
    
    normalized_data = clean_data.clone()

    for col in range(clean_data.shape[1]):
        scaler = MinMaxScaler(feature_range=(0, 1))
        normalized_data[:, col] = torch.tensor(scaler.fit_transform(clean_data[:, col].reshape(-1, 1)).flatten())

    for i in range(normalized_data.shape[1]):
        plt.subplot(2, 2, i + 1)
        plt.hist(normalized_data[:, i].numpy(), bins=30, alpha=0.7, label=features[i])
        plt.title(f'{features[i]} 分布')
        plt.xlabel('数值')
        plt.ylabel('频数')
        plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    DATA_PATH = 'data.xlsx'

    df = pd.read_excel(DATA_PATH, sheet_name="水泥行业原始数据")
    features = ["总排放量", "用电量(MWh)", "用烟煤量（吨）", "用柴油量（吨）"]

    df = df[features]
    plot_joint_results(df)