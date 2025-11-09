import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from tqdm import tqdm
import seaborn as sns

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
        5. 图数据生成
        """
        # 读取原始数据
        self.df = pd.read_excel(file_path, sheet_name="水泥行业原始数据")
        self.features = ["总排放量", "用电量(MWh)", "用烟煤量（吨）", "用柴油量（吨）"]
        self.clean_data = None
        self.normalized_data = None
        self.graph_data = None
        self.scalers = None

    def preprocess_data(self):
        """数据预处理流程"""
        # 处理缺失值
        self.df[self.features] = self.df[self.features].fillna(self.df[self.features].mean())

        # 强制非负处理
        self.df[self.features] = self.df[self.features].clip(lower=0)

        # 转换为Tensor并过滤异常值
        data_tensor = torch.tensor(self.df[self.features].values, dtype=torch.float32)

        # IQR异常值过滤
        for col in range(data_tensor.shape[1]):
            q1 = data_tensor[:, col].quantile(0.25)
            q3 = data_tensor[:, col].quantile(0.75)
            iqr = q3 - q1
            col_mask = (data_tensor[:, col] >= q1 - 1.5 * iqr) & (data_tensor[:, col] <= q3 + 1.5 * iqr)
            data_tensor = data_tensor[col_mask]

        self.clean_data = data_tensor

    def normalize_data(self):
        """数据归一化处理"""
        self.scalers = {}
        self.normalized_data = torch.zeros_like(self.clean_data)

        # 逐列归一化并保存scaler
        for col in range(self.clean_data.shape[1]):
            scaler = MinMaxScaler(feature_range=(0, 1))  # 明确设置范围
            col_data = self.clean_data[:, col].numpy().reshape(-1, 1)
            self.normalized_data[:, col] = torch.tensor(
                scaler.fit_transform(col_data).flatten(),
                dtype=torch.float32
            )
            self.scalers[col] = scaler

    def generate_graph_data(self):
        """生成图数据"""
        # 将数据转换为邻接矩阵形式
        n_samples = len(self.normalized_data)
        self.graph_data = torch.zeros((n_samples, 4, 4))

        # 基于特征相关性生成图结构
        for i in range(n_samples):
            sample = self.normalized_data[i]
            # 计算特征间的相关性作为边权重
            for j in range(4):
                for k in range(4):
                    if j != k:
                        correlation = torch.abs(sample[j] - sample[k])
                        self.graph_data[i, j, k] = correlation

    def prepare_data(self):
        """完整的数据准备流程"""
        self.preprocess_data()
        self.normalize_data()
        self.generate_graph_data()

    def __len__(self):
        return len(self.normalized_data)

    def __getitem__(self, idx):
        return self.normalized_data[idx], self.graph_data[idx]

# ==================== 生成对抗网络模型 ====================
class DataGenerator(nn.Module):
    def __init__(self, latent_dim=128):
        """
        数据生成器模型
        结构特点：
        - 使用Sigmoid输出层确保生成值在[0,1]范围
        - 增强深层网络结构以适应数据维度
        """
        super(DataGenerator, self).__init__()
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
            nn.Linear(256, 4),
            nn.Sigmoid()  # 确保输出在[0,1]范围
        )
    
    def forward(self, z):
        return self.model(z)

class DataDiscriminator(nn.Module):
    def __init__(self):
        """数据判别器模型"""
        super(DataDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 256),
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

class GraphGenerator(nn.Module):
    def __init__(self, latent_dim=128):
        """图生成器模型"""
        super(GraphGenerator, self).__init__()
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
            nn.Linear(256, 16),  # 4x4 邻接矩阵
            nn.Sigmoid()
        )
    
    def forward(self, z):
        x = self.model(z)
        return x.view(-1, 4, 4)  # 重塑为邻接矩阵形式

class GraphDiscriminator(nn.Module):
    def __init__(self):
        """图判别器模型"""
        super(GraphDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)  # 添加通道维度
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ==================== 训练模块 ====================
def compute_gradient_penalty(D, real_samples, fake_samples, device):
    """计算WGAN-GP梯度惩罚项"""
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1).to(device)
    
    # 处理三维数据的情况
    if len(real_samples.size()) == 3:
        alpha = alpha.unsqueeze(-1).expand(-1, real_samples.size(1), real_samples.size(2))
    else:
        # 确保fake_samples的维度与real_samples匹配
        if fake_samples.size(1) != real_samples.size(1):
            fake_samples = fake_samples.view(fake_samples.size(0), real_samples.size(1))
        alpha = alpha.expand_as(real_samples)
    
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    gradient_penalty = ((gradients.view(batch_size, -1).norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train_joint_gan(dataset, num_epochs=800, batch_size=64, latent_dim=128, device=DEVICE):
    """联合训练函数"""
    # 初始化模型
    data_generator = DataGenerator(latent_dim).to(device)
    data_discriminator = DataDiscriminator().to(device)
    graph_generator = GraphGenerator(latent_dim).to(device)
    graph_discriminator = GraphDiscriminator().to(device)
    
    # 优化器配置
    optim_DG = Adam(data_generator.parameters(), lr=0.0002, betas=(0.5, 0.9))
    optim_DD = Adam(data_discriminator.parameters(), lr=0.0002, betas=(0.5, 0.9))
    optim_GG = Adam(graph_generator.parameters(), lr=0.0002, betas=(0.5, 0.9))
    optim_GD = Adam(graph_discriminator.parameters(), lr=0.0002, betas=(0.5, 0.9))
    
    # 数据加载
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 训练记录
    losses = {'data_G': [], 'data_D': [], 'graph_G': [], 'graph_D': []}
    
    # 训练循环
    progress_bar = tqdm(range(num_epochs), desc="训练进度")
    for epoch in progress_bar:
        for real_data, real_graph in dataloader:
            real_data = real_data.to(device)
            real_graph = real_graph.to(device)
            batch_size = real_data.size(0)
            
            # 训练数据判别器
            for _ in range(5):
                optim_DD.zero_grad()
                
                real_validity = data_discriminator(real_data)
                d_loss_real = -torch.mean(real_validity)
                
                z = torch.randn(batch_size, latent_dim).to(device)
                with torch.no_grad():
                    fake_data = data_generator(z)
                fake_validity = data_discriminator(fake_data)
                d_loss_fake = torch.mean(fake_validity)
                
                gradient_penalty = compute_gradient_penalty(
                    data_discriminator, real_data, fake_data, device
                )
                
                d_loss = d_loss_real + d_loss_fake + 10 * gradient_penalty
                d_loss.backward()
                optim_DD.step()
            
            # 训练图判别器
            for _ in range(5):
                optim_GD.zero_grad()
                
                real_validity = graph_discriminator(real_graph)
                g_loss_real = -torch.mean(real_validity)
                
                z = torch.randn(batch_size, latent_dim).to(device)
                with torch.no_grad():
                    fake_graph = graph_generator(z)
                fake_validity = graph_discriminator(fake_graph)
                g_loss_fake = torch.mean(fake_validity)
                
                gradient_penalty = compute_gradient_penalty(
                    graph_discriminator, real_graph, fake_graph, device
                )
                
                gd_loss = g_loss_real + g_loss_fake + 10 * gradient_penalty
                gd_loss.backward()
                optim_GD.step()
            
            # 训练生成器
            optim_DG.zero_grad()
            optim_GG.zero_grad()
            
            z = torch.randn(batch_size, latent_dim).to(device)
            gen_data = data_generator(z)
            gen_graph = graph_generator(z)
            
            # 数据生成器损失
            dg_loss = -torch.mean(data_discriminator(gen_data))
            # 图生成器损失
            gg_loss = -torch.mean(graph_discriminator(gen_graph))
            
            # 联合损失
            g_loss = dg_loss + gg_loss
            g_loss.backward()
            
            optim_DG.step()
            optim_GG.step()
            
            # 记录损失
            losses['data_D'].append(d_loss.item())
            losses['data_G'].append(dg_loss.item())
            losses['graph_D'].append(gd_loss.item())
            losses['graph_G'].append(gg_loss.item())
        
        # 更新进度条
        progress_bar.set_postfix({
            'D Loss': f'{d_loss.item():.4f}',
            'G Loss': f'{g_loss.item():.4f}'
        })
    
    return data_generator, graph_generator, losses

# ==================== 可视化模块 ====================
def plot_joint_results(dataset, data_generator, graph_generator, losses, scalers, device=DEVICE):
    """结果可视化"""
    plt.figure(figsize=(15, 10))
    
    # 生成样本
    data_generator.eval()
    graph_generator.eval()
    with torch.no_grad():
        noise = torch.randn(500, 128).to(device)
        generated_data = data_generator(noise).cpu().numpy()
        generated_data_graph = graph_generator(noise).cpu().numpy()
    
    # 对generated_data_graph进行可视化
    adj_matrix = generated_data_graph[0]  # 取第一个生成的邻接矩阵样本
    feature_names = ["总排放量", "用电量(MWh)", "用烟煤量（吨）", "用柴油量（吨）"]  # 添加特征名称列表
    
    sns.heatmap(
        adj_matrix,
        annot=True,       
        fmt=".2f",        
        cmap="viridis",   
        cbar=True,        
        xticklabels=feature_names,  # 替换为真实特征名
        yticklabels=feature_names,  # 替换为真实特征名
        mask=np.diag(np.ones(4, dtype=bool))  # 隐藏对角线
    )
    plt.title("邻接矩阵热图：特征间相关性")
    plt.show()
    
    # 逆归一化数据
    real_data = dataset.clean_data.numpy()
    generated_data_original = np.zeros_like(generated_data)
    for col in range(4):
        generated_data_original[:, col] = scalers[col].inverse_transform(
            generated_data[:, col].reshape(-1, 1)
        ).flatten()
    
    # 绘制数据分布对比
    features = ["总排放量", "用电量(MWh)", "用烟煤量（吨）", "用柴油量（吨）"]
    for i, feature in enumerate(features):
        plt.subplot(2, 2, i + 1)
        plt.hist(real_data[:, i], bins=50, alpha=0.5, label='真实数据', color='blue')
        plt.hist(generated_data_original[:, i], bins=50, alpha=0.5, label='生成数据', color='orange')
        plt.title(f'{feature}分布对比')
        plt.xlabel('数值')
        plt.ylabel('频数')
        plt.legend()
    plt.tight_layout()
    plt.show()
    
    # 绘制损失曲线
    plt.subplot(2, 1, 1)
    plt.plot(losses['data_D'], label='数据判别器损失', alpha=0.7)
    plt.plot(losses['data_G'], label='数据生成器损失', alpha=0.7)
    plt.title('数据GAN训练损失')
    plt.xlabel('迭代次数')
    plt.ylabel('损失值')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(losses['graph_D'], label='图判别器损失', alpha=0.7)
    plt.plot(losses['graph_G'], label='图生成器损失', alpha=0.7)
    plt.title('图GAN训练损失')
    plt.xlabel('迭代次数')
    plt.ylabel('损失值')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 保存生成的特征数据
    np.savetxt('graph_generated_data.csv', generated_data, delimiter=',', header='总排放量,用电量(MWh),用烟煤量（吨）,用柴油量（吨）', comments='')
    # 计算均值和标准差并对比
    mean_fake = generated_data.mean(axis=0)
    std_fake = generated_data.std(axis=0)
    mean_real = real_data.mean(axis=0)
    std_real = real_data.std(axis=0)
    
    # 追加统计信息到CSV文件
    with open('graph_generated_data.csv', 'a') as f:
        f.write('\n# 均值和标准差对比\n')
        f.write('mean_fake,std_fake,mean_real,std_real\n')
        for i in range(4):
            f.write(f'{mean_fake[i]},{std_fake[i]},{mean_real[i]},{std_real[i]}\n')

# ==================== 主程序 ====================
if __name__ == "__main__":
    # 参数配置
    DATA_PATH = 'data.xlsx'  # 修改为实际数据路径
    
    # 数据预处理
    dataset = CementDataset(DATA_PATH)
    dataset.prepare_data()  # 使用统一的方法准备数据
    
    # 训练模型
    data_generator, graph_generator, losses = train_joint_gan(
        dataset=dataset,
        num_epochs=1000,
        batch_size=16,
        device=DEVICE
    )
    
    # 结果可视化
    plot_joint_results(dataset, data_generator, graph_generator, losses, dataset.scalers)