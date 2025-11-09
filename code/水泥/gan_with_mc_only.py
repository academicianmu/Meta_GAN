import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from tqdm import tqdm

# ==================== 全局配置 ====================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==================== 数据预处理模块 ====================
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
        self.df = pd.read_excel(file_path, sheet_name="造纸行业原始数据")
        self.features = ["总排放量（tCO2)"]

        # 数据清洗流程
        self._preprocess_data()
        self._normalize_data()

    def _preprocess_data(self):
        """数据预处理流程"""
        # 处理缺失值
        self.df[self.features] = self.df[self.features].fillna(self.df[self.features].mean())

        # 强制非负处理
        self.df[self.features] = self.df[self.features].clip(lower=0)

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

# ==================== 生成对抗网络模型 ====================
class Generator(nn.Module):
    def __init__(self, latent_dim=128):
        """
        生成器模型
        结构特点：
        - 使用Sigmoid输出层确保生成值在[0,1]范围
        - 增强深层网络结构以适应数据维度
        """
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()  # 确保输出在[0,1]范围
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self):
        """判别器模型（Critic）"""
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

# ==================== 训练模块 ====================
def compute_gradient_penalty(D, real_samples, fake_samples, device):
    """计算WGAN - GP梯度惩罚项"""
    alpha = torch.rand(real_samples.size(0), 1).to(device)
    target_shape = [1] * real_samples.dim()
    target_shape[0] = real_samples.size(0)
    alpha = alpha.view(target_shape).expand_as(real_samples)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

class ExperiencePool:
    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.pool = []

    def add(self, data):
        if len(self.pool) < self.max_size:
            self.pool.append(data)
        else:
            index = np.random.randint(0, self.max_size)
            self.pool[index] = data

    def sample(self, num_samples):
        if len(self.pool) < num_samples:
            num_samples = len(self.pool)
        indices = np.random.choice(len(self.pool), num_samples, replace=False)
        samples = [self.pool[i] for i in indices]
        return torch.stack(samples).to(DEVICE)

class MonteCarloEvaluator:
    def __init__(self, real_data, num_samples=1000, batch_size=16):
        """
        初始化评估器
        real_data: 真实数据集（numpy数组或torch张量）
        """
        self.num_samples = num_samples
        self.batch_size = batch_size
        
        # 转换为numpy数组
        if isinstance(real_data, torch.Tensor):
            real_data = real_data.cpu().numpy()
            
        self.real_data = real_data
        self.real_mean = np.mean(real_data, axis=0)
        self.real_std = np.std(real_data, axis=0)

    def evaluate(self, generator, latent_dim, device):
        scores = []
        generator.eval()
        with torch.no_grad():
            # 计算总批次数（向上取整）
            num_batches = (self.num_samples + self.batch_size - 1) // self.batch_size
            for _ in range(num_batches):
                z = torch.randn(self.batch_size, latent_dim).to(device)
                generated_data = generator(z).cpu().numpy()  # 转换为numpy数组评估
                for sample in generated_data:
                    score = self._evaluate_sample(sample)
                    scores.append(score)
        generator.train()
        return np.mean(scores)

    def _evaluate_sample(self, sample):
        """评估单个样本的质量（简化版）"""
        # 1. 非负性检查
        if np.any(sample < 0):
            return 0.0
        
        # 2. 分布相似度（Z-score方法）
        z_scores = np.abs((sample - self.real_mean) / (self.real_std + 1e-8))
        distribution_score = np.exp(-np.mean(z_scores))  # 越接近1表示越相似
        
        return distribution_score

# 在训练循环中集成经验池和蒙特卡洛评估
def train_gan(dataset, num_epochs=800, batch_size=16, latent_dim=128, device=DEVICE):
    """训练主函数"""
    # 初始化模型
    generator = Generator(latent_dim).to(device)
    discriminator = Discriminator().to(device)

    # 优化器配置
    optim_G = Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.9))
    optim_D = Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.9))

    # 数据加载
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 训练记录
    losses_G, losses_D = [], []

    # 经验池和蒙特卡洛评估器
    real_data_np = dataset.clean_data.numpy()
    experience_pool = ExperiencePool(max_size=1000)
    mc_evaluator = MonteCarloEvaluator(real_data=real_data_np, num_samples=100, batch_size=16)

    # 训练循环
    progress_bar = tqdm(range(num_epochs), desc="训练进度")
    for epoch in progress_bar:
        for real_data in dataloader:
            real_data = real_data.to(device)
            batch_size = real_data.size(0)

            # 训练判别器
            for _ in range(5):
                optim_D.zero_grad()

                # 真实数据损失
                real_validity = discriminator(real_data)
                d_loss_real = -torch.mean(real_validity)

                # 生成数据
                z = torch.randn(batch_size, latent_dim).to(device)
                with torch.no_grad():
                    fake_data = generator(z)
                fake_validity = discriminator(fake_data)
                d_loss_fake = torch.mean(fake_validity)

                # 梯度惩罚
                gradient_penalty = compute_gradient_penalty(discriminator, real_data, fake_data, device)

                # 总损失
                d_loss = d_loss_real + d_loss_fake + 10 * gradient_penalty
                d_loss.backward()
                optim_D.step()

            # 训练生成器
            optim_G.zero_grad()
            z = torch.randn(batch_size, latent_dim).to(device)
            gen_data = generator(z)
            g_loss = -torch.mean(discriminator(gen_data))
            g_loss.backward()
            optim_G.step()

            # 蒙特卡洛评估生成样本（每批次评估一次）
            mc_score = mc_evaluator.evaluate(generator, latent_dim, device)
            
            # 将高质量样本存入经验池（阈值可调整）
            if mc_score > 0.2:  # 降低阈值便于测试
                experience_pool.add(gen_data.cpu().detach())

            # 从经验池中采样并训练生成器（增强多样性）
            if len(experience_pool.pool) > batch_size:
                sampled_data = experience_pool.sample(batch_size)
                optim_G.zero_grad()
                g_loss_sampled = -torch.mean(discriminator(sampled_data))
                g_loss_sampled.backward()
                optim_G.step()

        # 记录损失（取最后一个批次的损失）
        losses_G.append(g_loss.item())
        losses_D.append(d_loss.item())

        # 更新进度条
        progress_bar.set_postfix({
            'D Loss': f'{d_loss.item():.4f}',
            'G Loss': f'{g_loss.item():.4f}',
            'MC Score': f'{mc_score:.4f}'  # 显示评估分数
        })

    return generator, losses_G, losses_D

# ==================== 可视化模块 ====================
def plot_results(dataset, generator, scalers, device=DEVICE):
    """结果可视化"""
    # 生成样本
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(500, 128).to(device)
        generated = generator(noise).cpu().numpy()

    # 逆归一化并确保非负
    generated_data = np.zeros_like(generated)
    for col in range(4):
        generated_data[:, col] = scalers[col].inverse_transform(generated[:, col].reshape(-1, 1)).flatten()
    generated_data = np.clip(generated_data, a_min=0, a_max=None)  # 强制非负

    # 可视化设置
    plt.figure(figsize=(14, 10))
    colors = ['#1f77b4', '#ff7f0e']

    # 总排放量分布对比
    plt.subplot(2, 2, 1)
    plt.hist(dataset.clean_data[:, 0].numpy(), bins=50, alpha=0.6, label='真实数据', color=colors[0])
    plt.hist(generated_data[:, 0], bins=50, alpha=0.6, label='生成数据', color=colors[1])
    plt.title('总排放量分布对比', fontsize=12, fontweight='bold')
    plt.xlabel('排放量（吨）', fontsize=10)
    plt.ylabel('频数', fontsize=10)
    plt.xlim(left=0)  # 强制x轴从0开始
    plt.legend()

    # 训练损失曲线
    plt.subplot(2, 1, 2)
    plt.plot(losses_D, label='判别器损失', color='#2ca02c')
    plt.plot(losses_G, label='生成器损失', color='#d62728')
    plt.title('训练损失曲线', fontsize=12, fontweight='bold')
    plt.xlabel('训练轮次', fontsize=10)
    plt.ylabel('损失值', fontsize=10)
    plt.legend()

    plt.tight_layout()
    plt.show()

# ==================== 主程序 ====================
if __name__ == "__main__":
    # 参数配置
    DATA_PATH = 'data.xlsx'  # 修改为实际数据路径

    # 数据预处理
    dataset = CementDataset(DATA_PATH)

    # 训练模型
    generator, losses_G, losses_D = train_gan(
        dataset=dataset,
        num_epochs=100,
        batch_size=64,
        device=DEVICE # 确保设备匹配
    )

    # 结果可视化
    plot_results(dataset, generator, dataset.scalers)