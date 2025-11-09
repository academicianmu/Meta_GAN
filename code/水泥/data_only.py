import torch
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class CementDataset(Dataset):
    def __init__(self, file_path):
        """
        水泥行业数据集处理类
        功能：
        1. 读取Excel数据
        2. 数据清洗和预处理
        3. 非负值强制处理
        4. 数据归一化
        """
        # 读取原始数据
        self.df = pd.read_excel(file_path, sheet_name="水泥行业原始数据")
        self.features = ["总排放量"]
        self.original_data = self.df[self.features].values.astype(float)
        self.original_mean = self.original_data.mean()
        self.original_std = self.original_data.std()
        self.original_min = self.original_data.min()
        self.original_max = self.original_data.max()
        self.original_number = len(self.original_data)
        
        self.clean_data = None

        # 数据清洗流程
        self._preprocess_data()
        self._normalize_data()

    def _preprocess_data(self):
        """数据预处理流程"""
        # 处理缺失值
        self.df[self.features] = self.df[self.features].fillna(self.df[self.features].mean())

        # 转换为Tensor并过滤异常值
        data_tensor = torch.tensor(self.df[self.features].values, dtype=torch.float32)

        # 双重过滤：先过滤所有负值，再进行IQR过滤
        mask = (data_tensor >= 0).all(dim=1)
        data_tensor = data_tensor[mask]

        # IQR异常值过滤
        for col in range(data_tensor.shape[1]):
            q1 = data_tensor[:, col].quantile(0.25)
            q3 = data_tensor[:, col].quantile(0.75)
            iqr = q3 - q1
            col_mask = (data_tensor[:, col] >= q1 - 1.5 * iqr) & (data_tensor[:, col] <= q3 + 1.5 * iqr)
            data_tensor = data_tensor[col_mask]

        self.clean_data = data_tensor

    def _normalize_data(self):
        """数据归一化处理"""
        self.scalers = {}
        self.normalized_data = self.clean_data.clone()

        # 逐列归一化并保存scaler
        for col in range(self.clean_data.shape[1]):
            scaler = MinMaxScaler(feature_range=(0, 1))  # 明确设置范围
            self.normalized_data[:, col] = torch.tensor(
                scaler.fit_transform(self.clean_data[:, col].reshape(-1, 1)).flatten()
            )
            self.scalers[col] = scaler

    def __len__(self):
        return len(self.normalized_data)

    def __getitem__(self, idx):
        return self.normalized_data[idx]

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CementDataset("data.xlsx")

    # 画图格线不明显
    plt.hist(dataset.normalized_data[:, 0].cpu().numpy(), bins=30, alpha=0.5, label="总排放量")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlabel("值")
    plt.ylabel("频率")
    plt.title("水泥行业排放数据直方图")
    plt.legend()
    plt.show()

    # 打印原始数据统计信息
    print(f"原始数据均值: {dataset.original_mean}, 标准差: {dataset.original_std}, 最小值: {dataset.original_min}, 最大值: {dataset.original_max}, 数量: {dataset.original_number}")
    
    # 打印清洗后的数据统计信息
    print(f"清洗后数据均值: {dataset.clean_data.mean().item()}, 标准差: {dataset.clean_data.std().item()}")
    print(f"清洗后数据最小值: {dataset.clean_data.min().item()}, 最大值: {dataset.clean_data.max().item()}")
    print(f"数据量: {len(dataset)}")

    # 打印归一化后的数据统计信息
    print(f"归一化后数据均值: {dataset.normalized_data.mean()}, 标准差: {dataset.normalized_data.std()}")
    print(f"归一化后数据最小值: {dataset.normalized_data.min()}, 最大值: {dataset.normalized_data.max()}")
    print(f"数据量: {len(dataset)}")