import time  # 导入时间模块，用于记录时间
import numpy as np  # 导入NumPy，用于数组和数值计算
import pandas as pd  # 导入Pandas，用于数据处理
import torch  # 导入PyTorch，用于深度学习
import torch.nn as nn  # 导入PyTorch的神经网络模块
import torch.optim as optim  # 导入PyTorch的优化器模块
from sklearn.preprocessing import StandardScaler  # 导入Sklearn的StandardScaler用于数据标准化
import matplotlib.pyplot as plt  # 导入Matplotlib用于绘图
from sklearn.model_selection import train_test_split  # 导入Sklearn的train_test_split用于数据集划分
from torch.utils.data import DataLoader, TensorDataset  # 导入PyTorch的数据加载工具
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error  # 导入Sklearn的评估指标
import matplotlib  # 导入Matplotlib的核心库
import math  # 导入数学库，用于数学运算
import torch.nn.functional as F  # 导入PyTorch的函数式接口
import csv  # 导入CSV模块，用于保存数据
import json  # 导入JSON模块，用于保存模型参数和评估结果
import logging  # 导入日志模块，用于记录训练过程
import argparse  # 导入argparse模块，用于解析命令行参数
import seaborn as sns  # 导入Seaborn用于更高级的可视化

# 解析命令行参数
parser = argparse.ArgumentParser(description="LSTM and KAN Model Training")  # 创建ArgumentParser对象
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')  # 添加学习率参数
parser.add_argument('--hidden_size', type=int, default=512, help='Hidden layer size for LSTM')  # 添加隐藏层大小参数
args = parser.parse_args()  # 解析命令行参数

learning_rate = args.learning_rate  # 从解析结果中获取学习率
hidden_layer_size = args.hidden_size  # 从解析结果中获取隐藏层大小

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 如果有GPU可用则使用，否则使用CPU
print(f"Using device: {device}")  # 输出正在使用的设备

# 设置字体为Times New Roman
matplotlib.rcParams['font.sans-serif'] = ['Times New Roman']  # 设置Matplotlib字体为Times New Roman
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

# 记录开始时间
start_time = time.time()  # 记录代码开始运行的时间

# 定义 KANLinear 类
class KANLinear(torch.nn.Module):  # 定义一个自定义的PyTorch模块类KANLinear
    def __init__(self, in_features, out_features, grid_size=5, spline_order=5, scale_noise=0.1, scale_base=1.0,
                 scale_spline=1.0, enable_standalone_scale_spline=True, base_activation=torch.nn.ReLU, grid_eps=0.02,
                 grid_range=[-1, 1]):  # 初始化函数，定义输入输出特征、网格大小、样条阶数等参数
        super(KANLinear, self).__init__()  # 调用父类的初始化方法
        self.in_features = in_features  # 输入特征的数量
        self.out_features = out_features  # 输出特征的数量
        self.grid_size = grid_size  # 栅格大小
        self.spline_order = spline_order  # 样条阶数

        h = (grid_range[1] - grid_range[0]) / grid_size  # 计算网格步长
        grid = (
            (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
            .expand(in_features, -1)
            .contiguous()
        )  # 创建并扩展网格
        self.register_buffer("grid", grid)  # 将网格注册为模型的缓冲区
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))  # 初始化基础权重参数
        self.spline_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))  # 初始化样条权重参数
        if enable_standalone_scale_spline:  # 如果启用了独立的样条缩放
            self.spline_scaler = torch.nn.Parameter(torch.Tensor(out_features, in_features))  # 初始化样条缩放器
        self.scale_noise = scale_noise  # 设置噪声比例
        self.scale_base = scale_base  # 设置基础比例
        self.scale_spline = scale_spline  # 设置样条比例
        self.enable_standalone_scale_spline = enable_standalone_scale_spline  # 是否启用独立样条缩放
        self.base_activation = base_activation()  # 设置基础激活函数
        self.grid_eps = grid_eps  # 设置网格间隔
        self.reset_parameters()  # 重置参数

    def reset_parameters(self):  # 重置参数
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)  # 使用Kaiming均匀初始化基础权重
        with torch.no_grad():  # 在不计算梯度的情况下
            noise = (
                (torch.rand(self.grid_size + 1, self.in_features,
                            self.out_features) - 1 / 2) * self.scale_noise / self.grid_size
            )  # 生成噪声
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0) * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order], noise)
            )  # 复制样条权重
            if self.enable_standalone_scale_spline:  # 如果启用了独立样条缩放
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)  # 初始化样条缩放器

    def b_splines(self, x: torch.Tensor):  # 计算B样条基函数
        assert x.dim() == 2 and x.size(1) == self.in_features  # 断言输入的维度和特征数匹配
        grid: torch.Tensor = (self.grid)  # 获取网格
        x = x.unsqueeze(-1)  # 在最后一个维度添加一个新维度
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)  # 计算样条基函数的初始值
        for k in range(1, self.spline_order + 1):  # 逐步计算样条基函数
            bases = (
                (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * bases[:, :, 1:]
            )

        assert bases.size() == (x.size(0), self.in_features, self.grid_size + self.spline_order)  # 断言样条基函数的尺寸匹配
        return bases.contiguous()  # 返回连续的样条基函数

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):  # 将曲线转换为系数
        assert x.dim() == 2 and x.size(1) == self.in_features  # 断言输入的维度和特征数匹配
        assert y.size() == (x.size(0), self.in_features, self.out_features)  # 断言输出的尺寸匹配
        A = self.b_splines(x).transpose(0, 1)  # 计算B样条基函数并转置
        B = y.transpose(0, 1)  # 转置输出
        solution = torch.linalg.lstsq(A, B).solution  # 计算最小二乘解
        result = solution.permute(2, 0, 1)  # 调整解的维度
        assert result.size() == (self.out_features, self.in_features, self.grid_size + self.spline_order)  # 断言结果的尺寸匹配
        return result.contiguous()  # 返回连续的结果

    @property
    def scaled_spline_weight(self):  # 定义一个属性返回缩放后的样条权重
        return self.spline_weight * (self.spline_scaler.unsqueeze(-1) if self.enable_standalone_scale_spline else 1.0)  # 根据是否启用独立样条缩放进行缩放

    def forward(self, x: torch.Tensor):  # 定义前向传播
        assert x.dim() == 2 and x.size(1) == self.in_features  # 断言输入的维度和特征数匹配
        base_output = F.linear(self.base_activation(x), self.base_weight)  # 计算基础输出
        spline_output = F.linear(self.b_splines(x).view(x.size(0), -1),
                                 self.scaled_spline_weight.view(self.out_features, -1))  # 计算样条输出
        return base_output + spline_output  # 返回基础输出和样条输出的和

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):  # 更新网格
        assert x.dim() == 2 and x.size(1) == self.in_features  # 断言输入的维度和特征数匹配
        batch = x.size(0)  # 获取批量大小
        splines = self.b_splines(x)  # 计算样条基函数
        splines = splines.permute(1, 0, 2)  # 调整维度
        orig_coeff = self.scaled_spline_weight  # 获取原始样条权重
        orig_coeff = orig_coeff.permute(1, 2, 0)  # 调整维度
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # 计算未简化的样条输出
        unreduced_spline_output = unreduced_spline_output.permute(1, 0, 2)  # 调整维度
        x_sorted = torch.sort(x, dim=0)[0]  # 对输入进行排序
        grid_adaptive = x_sorted[torch.linspace(0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device)]  # 计算自适应网格
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size  # 计算均匀步长
        grid_uniform = (
            torch.arange(self.grid_size + 1, dtype=torch.float32, device=x.device).unsqueeze(1) * uniform_step +
            x_sorted[0] - margin
        )  # 计算均匀网格
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive  # 结合均匀网格和自适应网格
        grid = torch.cat(
            [
                grid[:1] - uniform_step * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:] + uniform_step * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )  # 生成最终的网格
        self.grid.copy_(grid.T)  # 复制网格到缓冲区
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))  # 更新样条权重

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):  # 计算正则化损失
        l1_fake = self.spline_weight.abs().mean(-1)  # 计算L1损失
        regularization_loss_activation = l1_fake.sum()  # 计算激活函数正则化损失
        p = l1_fake / regularization_loss_activation  # 计算概率分布
        regularization_loss_entropy = -torch.sum(p * p.log())  # 计算熵正则化损失
        return regularize_activation * regularization_loss_activation + regularize_entropy * regularization_loss_entropy  # 返回总正则化损失

    def plot_activation_function(self, num_points=None, num_outputs_to_plot=3):  # 绘制激活函数
        if num_points is None:  # 如果未指定样本点数量
            num_points = self.grid_size + self.spline_order  # 使用默认样本点数量
        x = torch.linspace(self.grid[0, self.spline_order].item(), self.grid[0, -self.spline_order].item(),
                           num_points).unsqueeze(1)  # 生成输入数据
        x = x.expand(-1, self.in_features).to(self.grid.device)  # 扩展输入数据并移动到对应设备

        with torch.no_grad():  # 不计算梯度
            bases = self.b_splines(x)  # 计算样条基函数
            spline_weight_adjusted = self.spline_weight.permute(2, 1, 0)  # 调整样条权重的维度
            y = torch.einsum('bik,kjo->bio', bases, spline_weight_adjusted)  # 计算激活函数值
        plt.figure(figsize=(10, 6))  # 设置图形大小
        for i in range(min(self.out_features, num_outputs_to_plot)):  # 遍历要绘制的输出数量
            plt.plot(x[:, 0].cpu().numpy(), y[:, :, i].mean(dim=1).cpu().numpy(), label=f'Output {i + 1}')  # 绘制激活函数
        plt.title('Learned Activation Function')  # 设置图标题
        plt.xlabel('Input')  # 设置X轴标签
        plt.ylabel('Activation')  # 设置Y轴标签
        plt.legend()  # 显示图例
        plt.grid(True)  # 显示网格

class KAN(torch.nn.Module):  # 定义KAN模型类
    def __init__(self, layers_hidden, grid_size=5, spline_order=5, scale_noise=0.1, scale_base=1.0, scale_spline=1.0,
                 base_activation=torch.nn.SiLU, grid_eps=0.02, grid_range=[-1, 1]):  # 初始化函数
        super(KAN, self).__init__()  # 调用父类的初始化方法
        self.grid_size = grid_size  # 设置网格大小
        self.spline_order = spline_order  # 设置样条阶数
        self.layers = torch.nn.ModuleList()  # 初始化一个空的模块列表
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):  # 遍历隐藏层配置
            self.layers.append(
                KANLinear(in_features, out_features, grid_size=grid_size, spline_order=spline_order,
                          scale_noise=scale_noise, scale_base=scale_base, scale_spline=scale_spline,
                          base_activation=base_activation, grid_eps=grid_eps, grid_range=grid_range)
            )  # 添加KANLinear层

    def forward(self, x: torch.Tensor, update_grid=False):  # 定义前向传播
        for layer in self.layers:  # 遍历所有层
            if update_grid:  # 如果需要更新网格
                layer.update_grid(x)  # 更新当前层的网格
            x = layer(x)  # 前向传播
        return x  # 返回最终输出

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):  # 计算正则化损失
        return sum(layer.regularization_loss(regularize_activation, regularize_entropy) for layer in self.layers)  # 聚合所有层的正则化损失

# 数据导入
file_path = r"E:\111\Earthquake-Prediction-main\Processed data\5组处理后的数据\需要处理的数据-1\1-速度-modified.csv"  # 数据文件路径
data = pd.read_csv(file_path)  # 读取CSV文件
data.columns = ['Time'] + [f'acceleration_{i}' for i in range(1, len(data.columns))]  # 重命名列名

# 数据检查
print("Data shape:", data.shape)  # 打印数据形状
print("Data columns:", data.columns)  # 打印数据列名
print("Data summary:")
print(data.describe())  # 打印数据的统计信息

# 数据可视化
plt.figure(figsize=(10, 6))
plt.plot(data['Time'], data.iloc[:, 1], label='Acceleration Feature 1')  # 绘制特征随时间变化的图
plt.xlabel('Time')
plt.ylabel('Acceleration')
plt.title('Raw Data Overview')
plt.legend()
plt.grid(True)
plt.show()

# 数据预处理
data.iloc[:, 1:] = data.iloc[:, 1:].rolling(window=10, min_periods=1).mean()  # 对数据进行滑动平均处理
features = data.iloc[:, 1:].values  # 获取所有特征数据
scaler = StandardScaler()  # 初始化标准化器
scaled_features = scaler.fit_transform(features)  # 标准化特征数据

# 划分训练集和测试集（不打乱数据）
train_size = int(len(scaled_features) * 0.7)  # 计算训练集大小（70%）
train_features = scaled_features[:train_size]  # 获取训练集数据
test_features = scaled_features[train_size:]  # 获取测试集数据

# 数据增强
def augment_data(features):
    noise = np.random.normal(0, 0.1, features.shape)  # 生成正态分布的噪声
    augmented_features = features + noise  # 添加噪声
    return augmented_features

augmented_train_features = augment_data(train_features)  # 对训练集进行数据增强

# 准备训练数据
def create_dataset(features, time_step=20):  # 定义函数创建时间序列数据集
    dataX, dataY = [], []  # 初始化空列表
    for i in range(len(features) - time_step):  # 遍历所有数据
        dataX.append(features[i:(i + time_step)])  # 获取时间步长的输入
        dataY.append(features[i + time_step])  # 获取对应的输出
    return np.array(dataX), np.array(dataY)  # 返回数组形式的输入和输出

time_step = 20  # 设置时间步长
X_train, y_train = create_dataset(augmented_train_features, time_step)  # 创建训练数据集
X_test, y_test = create_dataset(test_features, time_step)  # 创建测试数据集

# 转换为张量
X_train = torch.from_numpy(X_train).float().to(device)  # 转换训练输入为PyTorch张量并移动到设备
y_train = torch.from_numpy(y_train).float().to(device)  # 转换训练输出为PyTorch张量并移动到设备
X_test = torch.from_numpy(X_test).float().to(device)  # 转换测试输入为PyTorch张量并移动到设备
y_test = torch.from_numpy(y_test).float().to(device)  # 转换测试输出为PyTorch张量并移动到设备

# 创建LSTM和KAN模型的组合
class LSTM_KAN_Model(nn.Module):  # 定义LSTM和KAN的组合模型
    def __init__(self, input_size, output_size, hidden_layer_size=512, num_layers=2, dropout=0.5):  # 初始化函数
        super(LSTM_KAN_Model, self).__init__()  # 调用父类的初始化方法
        self.lstm1 = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True, dropout=dropout)  # 第一个2层LSTM
        layers_hidden_1 = [hidden_layer_size, 512]  # 第一层 KAN的隐藏层配置
        self.kan1 = KAN(layers_hidden=layers_hidden_1)  # 创建第一个KAN层
        self.lstm2 = nn.LSTM(512, hidden_layer_size, num_layers, batch_first=True, dropout=dropout)  # 第二个2层LSTM
        layers_hidden_2 = [hidden_layer_size, output_size]  # 第二层 KAN的输出层配置
        self.kan2 = KAN(layers_hidden=layers_hidden_2)  # 创建第二个KAN层

    def forward(self, input_seq):  # 定义前向传播
        lstm_out1, _ = self.lstm1(input_seq)  # 通过第一个LSTM
        lstm_out1 = lstm_out1[:, -1, :]  # 提取最后一个时间步的输出
        kan_out1 = self.kan1(lstm_out1)  # 通过第一个KAN
        lstm_out2, _ = self.lstm2(kan_out1.unsqueeze(1))  # 通过第二个LSTM（需扩展维度以匹配输入格式）
        lstm_out2 = lstm_out2[:, -1, :]  # 提取最后一个时间步的输出
        kan_out2 = self.kan2(lstm_out2)  # 通过第二个KAN
        return kan_out2  # 返回最终输出

input_size = X_train.shape[2]  # 输入特征的数量
output_size = y_train.shape[1]  # 输出特征的数量
model = LSTM_KAN_Model(input_size=input_size, output_size=output_size, hidden_layer_size=hidden_layer_size).to(device)  # 创建模型并移动到设备

# 设置日志
logging.basicConfig(filename='training.log', level=logging.INFO)  # 配置日志记录

# 训练模型
loss_function = nn.MSELoss()  # 定义均方误差损失函数
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # 定义Adam优化器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)  # 定义学习率调度器
batch_size = 64  # 批量大小
train_data = TensorDataset(X_train, y_train)  # 创建训练数据集
train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size, drop_last=True)  # 创建训练数据加载器
test_data = TensorDataset(X_test, y_test)  # 创建测试数据集
test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size, drop_last=True)  # 创建测试数据加载器

epochs = 1500  # 训练轮次
patience = 100  # 提前停止的耐心值
best_loss = float('inf')  # 最佳损失初始化为无穷大
patience_counter = 0  # 耐心计数器
train_losses = []  # 保存训练损失
test_losses = []  # 保存测试损失

for epoch in range(epochs):  # 开始训练循环
    model.train()  # 将模型设置为训练模式
    epoch_start_time = time.time()  # 记录每个epoch开始时间
    epoch_train_loss = 0  # 初始化每轮训练损失
    for seq, labels in train_loader:  # 遍历训练数据
        optimizer.zero_grad()  # 梯度清零
        y_pred = model(seq)  # 预测
        single_loss = loss_function(y_pred, labels)  # 计算损失
        single_loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        epoch_train_loss += single_loss.item()  # 累积损失
    epoch_train_loss /= len(train_loader)  # 平均每轮损失
    train_losses.append(epoch_train_loss)  # 记录训练损失

    model.eval()  # 将模型设置为评估模式
    epoch_test_loss = 0  # 初始化每轮测试损失
    with torch.no_grad():  # 禁用梯度计算
        for seq, labels in test_loader:  # 遍历测试数据
            y_pred = model(seq)  # 预测
            single_loss = loss_function(y_pred, labels)  # 计算损失
            epoch_test_loss += single_loss.item()  # 累积损失
    epoch_test_loss /= len(test_loader)  # 平均每轮测试损失
    test_losses.append(epoch_test_loss)  # 记录测试损失

    epoch_end_time = time.time()  # 记录每个epoch结束时间
    logging.info(f"Epoch {epoch}: Train Loss = {epoch_train_loss}, Test Loss = {epoch_test_loss}, Time = {epoch_end_time - epoch_start_time:.2f}s")  # 记录日志

    scheduler.step(epoch_train_loss)  # 更新学习率
    if epoch % 25 == 0:  # 每25轮打印一次
        print(f'Epoch: {epoch} Train Loss: {epoch_train_loss} Test Loss: {epoch_test_loss}')  # 打印当前轮次的损失
        torch.save(model.state_dict(), f'model_epoch_{epoch}.pth')  # 每25个epoch保存一次模型

    if epoch_train_loss < best_loss:  # 如果当前损失是最佳损失
        best_loss = epoch_train_loss  # 更新最佳损失
        patience_counter = 0  # 重置耐心计数器
        torch.save(model.state_dict(), 'best_lstm_kan_model.pth')  # 保存最佳模型
    else:
        patience_counter += 1  # 增加耐心计数器

    if patience_counter >= patience:  # 如果耐心计数器达到上限
        print("Early stopping")  # 打印提前停止
        break  # 停止训练

model.load_state_dict(torch.load('best_lstm_kan_model.pth'))  # 加载最佳模型

# 预测
model.eval()  # 设置模型为评估模式
train_predictions = []  # 初始化训练集预测结果列表
test_predictions = []  # 初始化测试集预测结果列表
with torch.no_grad():  # 不计算梯度
    train_data_loader = DataLoader(TensorDataset(X_train), batch_size=batch_size, shuffle=False)  # 创建训练数据加载器
    test_data_loader = DataLoader(TensorDataset(X_test), batch_size=batch_size, shuffle=False)  # 创建测试数据加载器

    for seq in train_data_loader:  # 遍历训练数据
        seq = seq[0]  # 获取数据
        train_predictions.append(model(seq).cpu().numpy())  # 预测并保存训练集结果
    for seq in test_data_loader:  # 遍历测试数据
        seq = seq[0]  # 获取数据
        test_predictions.append(model(seq).cpu().numpy())  # 预测并保存测试集结果

train_predictions = np.concatenate(train_predictions, axis=0)  # 拼接训练集预测结果
test_predictions = np.concatenate(test_predictions, axis=0)  # 拼接测试集预测结果

# 逆标准化
train_predictions = scaler.inverse_transform(train_predictions)  # 逆标准化训练集预测结果
test_predictions = scaler.inverse_transform(test_predictions)  # 逆标准化测试集预测结果
y_train = scaler.inverse_transform(y_train.cpu().numpy())  # 逆标准化训练集真实值
y_test = scaler.inverse_transform(y_test.cpu().numpy())  # 逆标准化测试集真实值

# 保存预测结果到CSV文件
with open('train_predictions.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Index'] + [f'Predicted_Train_Feature_{i + 1}' for i in range(output_size)])  # 写入列名
    for idx, row in enumerate(train_predictions):
        writer.writerow([idx] + list(row))

with open('test_predictions.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Index'] + [f'Predicted_Test_Feature_{i + 1}' for i in range(output_size)])  # 写入列名
    for idx, row in enumerate(test_predictions):
        writer.writerow([idx] + list(row))

print("Predictions saved to CSV files: 'train_predictions.csv' and 'test_predictions.csv'.")  # 打印CSV保存成功

# 模型评价
train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))  # 计算训练集RMSE
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))  # 计算测试集RMSE
train_mae = mean_absolute_error(y_train, train_predictions)  # 计算训练集MAE
test_mae = mean_absolute_error(y_test, test_predictions)  # 计算测试集MAE
train_r2 = r2_score(y_train, train_predictions)  # 计算训练集R²
test_r2 = r2_score(y_test, test_predictions)  # 计算测试集R²

# 打印所有评估指标
print(f'Training Set RMSE: {train_rmse}, Training Set MAE: {train_mae}, Training Set R²: {train_r2}')
print(f'Testing Set RMSE: {test_rmse}, Testing Set MAE: {test_mae}, Testing Set R²: {test_r2}')

# 保存模型参数和评估结果到JSON文件
results = {
    'learning_rate': float(learning_rate),
    'hidden_layer_size': int(hidden_layer_size),
    'train_rmse': float(train_rmse),
    'test_rmse': float(test_rmse),
    'train_mae': float(train_mae),
    'test_mae': float(test_mae),
    'train_r2': float(train_r2),
    'test_r2': float(test_r2),
}

with open('results.json', 'w') as f:
    json.dump(results, f)

print("Model results and parameters saved to 'results.json'.")  # 打印JSON保存成功

# 绘制学习曲线
plt.figure(figsize=(14, 8))
plt.rc('font', family='Times New Roman')
plt.plot(train_losses, label='Training Loss', linewidth=4, color='r')
plt.plot(test_losses, label='Testing Loss', linewidth=4, color='k')
plt.xlabel('Epoch', fontsize=25)
plt.ylabel('Loss', fontsize=25)
plt.title('Learning Curve', fontsize=25)
plt.legend(fontsize=24, frameon=False)
plt.grid(True)
plt.savefig('Learning_Curve.png')



#可视化结果并输出各个点的坐标
for i in range(output_size):  # 遍历所有输出特征
    plt.figure #8203;:contentReference[oaicite:0]{index=0}&#8203;
    plt.figure(figsize=(14, 8))  # 设置图形大小
    plt.rc('font', family='Times New Roman')
    line_width = 4  # 设置线条宽度
    font_size = 26  # 设置字体大小

    total_length = len(y_test) + len(y_train)  # 计算训练集和测试集总长度

    # 存储坐标数据的列表
    train_pred_coords = list(zip(range(len(y_train)), train_predictions[:, i]))
    train_actual_coords = list(zip(range(len(y_train)), y_train[:, i]))
    test_pred_coords = list(zip(range(len(y_train), len(y_train) + len(y_test)), test_predictions[:, i]))
    test_actual_coords = list(zip(range(len(y_train), len(y_train) + len(y_test)), y_test[:, i]))

    # 绘制训练集预测结果
    plt.plot(range(len(y_train)), train_predictions[:, i], label=f'Predicted Training Feature_{i + 1}', color='orange', linewidth=line_width)
    plt.plot(range(len(y_train)), y_train[:, i], label=f'Actual Training Feature_{i + 1}', color='blue', linewidth=4)

    # 绘制测试集预测结果
    plt.plot(range(len(y_train), len(y_train) + len(y_test)), test_predictions[:, i], label=f'Predicted Testing Feature_{i + 1}', color='cyan', linewidth=line_width)
    plt.plot(range(len(y_train), len(y_train) + len(y_test)), y_test[:, i], label=f'Actual Testing Feature_{i + 1}', color='green', linewidth=4)

    # 设置图形参数
    plt.xlabel('Time', fontsize=font_size)  # 设置X轴标签和字体大小
    plt.ylabel(f'Feature_{i + 1}', fontsize=font_size)  # 设置Y轴标签和字体大小
    plt.tick_params(axis='both', which='major', labelsize=25, width=3)  # 加粗刻度线宽度为3
    for spine in plt.gca().spines.values():
        spine.set_linewidth(3)  # 边框线条加粗
    plt.legend(fontsize=font_size, frameon=False)  # 显示图例并设置字体大小
    plt.savefig(f'Feature_{i + 1}.png')  # 保存图形为PNG文件
    plt.tight_layout()  # 自动调整子图参数
    plt.show()  # 显示图形

# 绘制残差图
for i in range(output_size):
    plt.figure(figsize=(14, 8))
    plt.rc('font', family='Times New Roman')
    residuals_train = y_train[:, i] - train_predictions[:, i]
    residuals_test = y_test[:, i] - test_predictions[:, i]
    plt.scatter(range(len(y_train)), residuals_train, label=f'Residuals Train Feature_{i + 1}', color='blue', alpha=0.5)
    plt.scatter(range(len(y_train), len(y_train) + len(y_test)), residuals_test, label=f'Residuals Test Feature_{i + 1}', color='red', alpha=0.5)
    plt.axhline(0, color='black', linewidth=1)
    plt.xlabel('Index', fontsize=26)
    plt.ylabel(f'Residual Feature_{i + 1}', fontsize=26)
    plt.title(f'Residual Plot for Feature_{i + 1}', fontsize=26)
    plt.legend(fontsize=20, frameon=False)
    plt.grid(True)
    plt.savefig(f'Residual_Plot_Feature_{i + 1}.png')

# 绘制误差分布图
for i in range(output_size):
    plt.figure(figsize=(14, 8))
    plt.rc('font', family='Times New Roman')
    train_errors = y_train[:, i] - train_predictions[:, i]
    test_errors = y_test[:, i] - test_predictions[:, i]
    plt.hist(train_errors, bins=30, alpha=0.5, color='blue', label=f'Train Errors Feature_{i + 1}')
    plt.hist(test_errors, bins=30, alpha=0.5, color='red', label=f'Test Errors Feature_{i + 1}')
    plt.xlabel('Error', fontsize=26)
    plt.ylabel('Frequency', fontsize=26)
    plt.title(f'Error Distribution for Feature_{i + 1}', fontsize=26)
    plt.legend(fontsize=20, frameon=False)
    plt.grid(True)
    plt.savefig(f'Error_Distribution_Feature_{i + 1}.png')

# 绘制差异热图
for i in range(output_size):
    plt.figure(figsize=(14, 8))
    diff = y_test[:, i] - test_predictions[:, i]
    sns.heatmap(diff.reshape(-1, 1), cmap='coolwarm', annot=True, cbar=True)
    plt.title(f'Heatmap of Prediction Differences for Feature_{i + 1}')
    plt.savefig(f'Heatmap_Feature_{i + 1}.png')

# 绘制第一层 KAN 的激活函数变化图
model.kan2.layers[0].plot_activation_function()

# 记录结束时间并计算运行时间
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time:.2f} seconds")
