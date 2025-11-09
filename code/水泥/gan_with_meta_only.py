import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==================== 数据预处理模块 ====================
class CementDataset(Dataset):
    def __init__(self, file_path):
        self.df = pd.read_excel(file_path, sheet_name="水泥行业原始数据")
        self.features = ["总排放量"]
        self.scaler = MinMaxScaler(feature_range=(-1, 1))  
        self.data = self._process_data()

    def _process_data(self):
        df_clean = self.df[self.features].fillna(self.df[self.features].mean()).clip(lower=0)
        
        # 对数变换
        for feature in self.features:
            df_clean[feature] = np.log1p(df_clean[feature])  
        
        # Z-Score 过滤
        for feature in self.features:
            mean = df_clean[feature].mean()
            std = df_clean[feature].std()
            df_clean = df_clean[(df_clean[feature] > mean - 3*std) & (df_clean[feature] < mean + 3*std)]
        
        # IQR 二次过滤
        for feature in self.features:
            Q1 = df_clean[feature].quantile(0.25)
            Q3 = df_clean[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df_clean = df_clean[(df_clean[feature] >= lower) & (df_clean[feature] <= upper)]
        
        data = self.scaler.fit_transform(df_clean)
        return torch.tensor(data, dtype=torch.float32, device=DEVICE)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# ==================== 基础 GAN 模型 ====================
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# ==================== 基础元学习训练（简化版） ====================
def meta_train_gan(dataset, num_meta_epochs=200, num_inner_epochs=3, 
                   batch_size=64, latent_dim=100, meta_lr=0.0001, inner_lr=0.00005):
    G = Generator(latent_dim).to(DEVICE)
    D = Discriminator().to(DEVICE)
    meta_optim_G = optim.Adam(G.parameters(), lr=meta_lr)
    meta_optim_D = optim.Adam(D.parameters(), lr=meta_lr)
    criterion = nn.BCELoss()

    meta_g_losses = []
    meta_d_losses = []

    def sample_meta_task(dataset, task_size=batch_size):
        indices = np.random.choice(len(dataset), size=task_size, replace=False)
        return dataset.data[indices]

    meta_epoch_bar = tqdm(range(num_meta_epochs), desc="元训练轮次", unit="meta_epoch")
    for meta_epoch in meta_epoch_bar:
        task_g_losses = []
        task_d_losses = []

        for _ in range(5):  
            real_data_task = sample_meta_task(dataset)
            task_dataloader = DataLoader(TensorDataset(real_data_task), 
                                         batch_size=batch_size, shuffle=True)

            # 备份原始参数
            old_G_params = [param.clone() for param in G.parameters()]
            old_D_params = [param.clone() for param in D.parameters()]

            inner_bar = tqdm(task_dataloader, desc=f"Meta-Epoch {meta_epoch+1} Inner", 
                             unit="batch", leave=False)
            for i, batch in enumerate(inner_bar):
                real_data = batch[0].to(DEVICE)
                batch_size = real_data.size(0)
                real_labels = torch.ones(batch_size, 1, device=DEVICE)
                fake_labels = torch.zeros(batch_size, 1, device=DEVICE)

                # 内层训练判别器
                D.zero_grad()
                d_output_real = D(real_data)
                d_loss_real = criterion(d_output_real, real_labels)
                z = torch.randn(batch_size, latent_dim, device=DEVICE)
                fake_data = G(z).detach()
                d_output_fake = D(fake_data)
                d_loss_fake = criterion(d_output_fake, fake_labels)
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                # 手动更新判别器参数（内层更新）
                for param in D.parameters():
                    if param.grad is not None:
                        param.data -= inner_lr * param.grad

                # 内层训练生成器
                G.zero_grad()
                g_output = D(G(z))
                g_loss = criterion(g_output, real_labels)
                g_loss.backward()
                # 手动更新生成器参数（内层更新）
                for param in G.parameters():
                    if param.grad is not None:
                        param.data -= inner_lr * param.grad

                if i >= num_inner_epochs - 1:
                    break  

            # 收集内层最后一轮损失
            z = torch.randn(batch_size, latent_dim, device=DEVICE)
            fake_data = G(z)
            g_output = D(fake_data)
            final_g_loss = criterion(g_output, real_labels)
            d_output_real = D(real_data)
            final_d_loss = criterion(d_output_real, real_labels) + criterion(D(fake_data.detach()), fake_labels)
            
            task_g_losses.append(final_g_loss)
            task_d_losses.append(final_d_loss)

            # 恢复参数，准备下一个子任务
            for param, old_param in zip(G.parameters(), old_G_params):
                param.data = old_param.data
            for param, old_param in zip(D.parameters(), old_D_params):
                param.data = old_param.data

        # 元更新
        meta_g_loss = torch.stack(task_g_losses).mean()
        meta_d_loss = torch.stack(task_d_losses).mean()

        # 生成器元更新
        meta_optim_G.zero_grad()
        meta_g_loss.backward()
        meta_optim_G.step()

        # 判别器元更新
        meta_optim_D.zero_grad()
        meta_d_loss.backward()
        meta_optim_D.step()

        # 记录损失
        meta_g_losses.extend([loss.item() for loss in task_g_losses])
        meta_d_losses.extend([loss.item() for loss in task_d_losses])
        meta_epoch_bar.set_postfix({
            "Meta G Loss": f"{meta_g_loss.item():.4f}",
            "Meta D Loss": f"{meta_d_loss.item():.4f}"
        })

    return G, meta_g_losses, meta_d_losses

# ==================== 可视化模块 ====================
def plot_results(dataset, generator, g_losses, d_losses):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(500, 100, device=DEVICE)
        fake_data = generator(z).cpu().numpy()

    real_data = dataset.data.cpu().numpy()
    fake_data_original = np.expm1(dataset.scaler.inverse_transform(fake_data))
    real_data_original = np.expm1(dataset.scaler.inverse_transform(real_data))

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.hist(real_data_original[:, 0], bins=30, alpha=0.5, label='真实数据', density=True)
    plt.hist(fake_data_original[:, 0], bins=30, alpha=0.5, label='生成数据', density=True)
    plt.xlabel('数值')
    plt.ylabel('概率密度')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(g_losses, label='生成器损失', color='blue')
    plt.plot(d_losses, label='判别器损失', color='red')
    plt.title('训练损失曲线')
    plt.xlabel('迭代次数')
    plt.ylabel('损失值')
    plt.legend()
    plt.show()

# ==================== 主程序 ====================
if __name__ == "__main__":
    DATA_PATH = 'data.xlsx'
    dataset = CementDataset(DATA_PATH)

    # 元学习训练（基础版）
    generator_meta, g_losses_meta, d_losses_meta = meta_train_gan(
        dataset, num_meta_epochs=6000, num_inner_epochs=3, 
        batch_size=16, latent_dim=100, meta_lr=0.0001, inner_lr=0.0001
    )
    
    # 可视化（若微调则替换参数）
    plot_results(dataset, generator_meta, g_losses_meta, d_losses_meta)