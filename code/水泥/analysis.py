import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

def plot_joint_results(dataset):
    # plt.figure(figsize=(15, 10))
    for i, feature in enumerate(dataset.columns):
        # plt.subplot(2, 2, i+1)
        plt.hist(dataset[feature], bins=30, alpha=0.7, label=feature)
        plt.title(f'{feature} 分布')
        plt.xlabel('数值')
        plt.ylabel('频数')
        plt.legend()
            
    plt.show()
    
if __name__ == "__main__":
    DATA_PATH = 'data.xlsx'

    df = pd.read_excel(DATA_PATH, sheet_name="水泥行业原始数据")
    # features = ["总排放量", "用电量(MWh)", "用烟煤量（吨）", "用柴油量（吨）"]
    features = ["总排放量"]
    
    df = df[features]
    plot_joint_results(df)
