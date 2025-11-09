import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from tqdm import tqdm
import numpy as np

# ==================== 全局配置 ====================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==================== 数据预处理模块 ====================
class CementDataset(Dataset):
    def __init__(self, file_path):
        self.df = pd.read_excel(file_path, sheet_name="水泥行业原始数据")
        self.features = ["总排放量", "用电量(MWh)", "用烟煤量（吨）", "用柴油量（吨）"]
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = self._process_data()

    def _process_data(self):
        df_clean = self.df[self.features].fillna(self.df[self.features].mean()).clip(lower=0)
        data = self.scaler.fit_transform(df_clean)
        return torch.tensor(data, dtype=torch.float32, device=DEVICE)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# ==================== 基础GAN模型 ====================
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128), 
            nn.ReLU(),
            nn.Linear(128, 256), 
            nn.ReLU(),
            nn.Linear(256, 512), 
            nn.ReLU(),
            nn.Linear(512, 4), 
            nn.Sigmoid()
        )
    
    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 512), 
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256), 
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1), 
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# ========================= 训练模块 =========================
def train_gan(dataset, num_epochs=2000, batch_size=64, latent_dim=100):
    G = Generator(latent_dim).to(DEVICE)
    D = Discriminator().to(DEVICE)
    criterion = nn.BCELoss()
    optim_G = Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.9))
    optim_D = Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.9))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    g_losses = []
    d_losses = []
    
    # 主进度条（epoch层面）
    epoch_bar = tqdm(range(num_epochs), desc="训练轮次", unit="epoch")
    for epoch in epoch_bar:
        # 子进度条（batch层面）
        batch_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        for i, real_data in enumerate(batch_bar):
            batch_size = real_data.size(0)
            real_labels = torch.ones(batch_size, 1, device=DEVICE)
            fake_labels = torch.zeros(batch_size, 1, device=DEVICE)
            
            # 训练判别器
            D.zero_grad()
            d_output_real = D(real_data)
            d_loss_real = criterion(d_output_real, real_labels)
            z = torch.randn(batch_size, latent_dim, device=DEVICE)
            fake_data = G(z).detach()
            d_output_fake = D(fake_data)
            d_loss_fake = criterion(d_output_fake, fake_labels)
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optim_D.step()
            
            # 训练生成器
            G.zero_grad()
            g_output = D(G(z))
            g_loss = criterion(g_output, real_labels)
            g_loss.backward()
            optim_G.step()
            
            # 记录损失并更新进度条
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
            # batch_bar.set_postfix({  # 实时显示损失
            #     "D Loss": f"{d_loss.item():.4f}",
            #     "G Loss": f"{g_loss.item():.4f}"
            # })
        
        # 主进度条更新（每轮结束）
        epoch_bar.set_postfix({
            "Last D Loss": f"{d_loss.item():.4f}",
            "Last G Loss": f"{g_loss.item():.4f}"
        })
    
    return G, g_losses, d_losses

# ==================== 可视化模块 ====================
def plot_results(dataset, generator, g_losses, d_losses):
    # 生成样本
    generator.eval()
    with torch.no_grad():
        z = torch.randn(500, 100, device=DEVICE)
        fake_data = generator(z).cpu().numpy()
    
    # 逆归一化
    real_data = dataset.data.cpu().numpy()
    fake_data_original = dataset.scaler.inverse_transform(fake_data)
    real_data_original = dataset.scaler.inverse_transform(real_data)
    
    # 绘制分布对比
    features = ["总排放量", "用电量(MWh)", "用烟煤量（吨）", "用柴油量（吨）"]
    plt.figure(figsize=(12, 8))
    
    for i, feature in enumerate(features):
        plt.subplot(2, 2, i+1)
        plt.hist(real_data_original[:, i], bins=30, alpha=0.5, label='真实数据')
        plt.hist(fake_data_original[:, i], bins=30, alpha=0.5, label='生成数据')
        plt.title(f'{feature}分布对比')
        plt.xlabel('数值')
        plt.ylabel('概率密度')
        plt.legend()
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 4))
    plt.plot(g_losses, label='生成器损失', color='blue')
    plt.plot(d_losses, label='判别器损失', color='red')
    plt.title('训练损失曲线')
    plt.xlabel('迭代次数')
    plt.ylabel('损失值')
    plt.legend()
    plt.show()
    
    # 保存生成的特征数据
    np.savetxt('generated_data.csv', fake_data_original, delimiter=',', header='总排放量,用电量(MWh),用烟煤量（吨）,用柴油量（吨）', comments='')
    # 计算均值和标准差并对比
    mean_fake = fake_data_original.mean(axis=0)
    std_fake = fake_data_original.std(axis=0)
    mean_real = real_data_original.mean(axis=0)
    std_real = real_data_original.std(axis=0)
    
    # 追加统计信息到CSV文件
    with open('generated_data.csv', 'a') as f:
        f.write('\n# 均值和标准差对比\n')
        f.write('mean_fake,std_fake,mean_real,std_real\n')
        for i in range(4):
            f.write(f'{mean_fake[i]},{std_fake[i]},{mean_real[i]},{std_real[i]}\n')

# ==================== 主程序 ====================
if __name__ == "__main__":
    DATA_PATH = 'data.xlsx'
    dataset = CementDataset(DATA_PATH)
    generator, g_losses, d_losses = train_gan(dataset, num_epochs=20000, batch_size=64)
    plot_results(dataset, generator, g_losses, d_losses)