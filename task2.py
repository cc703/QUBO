import torch
import torch.nn as nn
import numpy as np
import kaiwu
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. 加载MNIST数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# 2. 构建CNN模型
class CNN(nn.Module):
    def __init__(self, conv1_out_channels=32, conv2_out_channels=64, fc1_out_features=128):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, conv1_out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(conv1_out_channels, conv2_out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(conv2_out_channels * 7 * 7, fc1_out_features)
        self.fc2 = nn.Linear(fc1_out_features, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # 除去batch维度的所有维度
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# 3. QUBO目标函数（优化CNN超参数）
def qubo_model(params):
    """
    QUBO目标函数，用于计算当前超参数集的能量值
    参数：params - 当前超参数
    返回：目标函数值（能量值）
    """
    # 将参数转换为整数形式
    conv1_out_channels = int(params[0])  # 假设是离散化后的值（例如 32, 64, 128）
    conv2_out_channels = int(params[1])
    fc1_out_features = int(params[2])

    # 正则化项：超参数的平方和
    energy_reg = np.sum(np.square([conv1_out_channels, conv2_out_channels, fc1_out_features]))

    # 交互项：卷积层输出通道数与全连接层之间的交互（这里只是示例）
    energy_interaction = (conv1_out_channels * fc1_out_features * 0.5) + (conv2_out_channels * fc1_out_features * 0.5)

    # 总能量：正则化项 + 交互项
    total_energy = energy_reg + energy_interaction
    return total_energy


# 4. 构建QUBO矩阵
def build_qubo_matrix(params):
    """
    将目标函数转换为QUBO矩阵
    """
    n = len(params)  # 超参数的数量
    Q = np.zeros((n, n))  # 初始化QUBO矩阵

    # 填充Q矩阵（这里只是一个简单示例，实际中需要根据模型的交互项填充Q矩阵）
    for i in range(n):
        Q[i, i] = 1  # 自身的二次项

    # 交互项（根据需求填充）
    Q[0, 1] = 0.5  # 示例：conv1_out_channels与conv2_out_channels之间的交互
    Q[1, 2] = 0.3  # 示例：conv2_out_channels与fc1_out_features之间的交互

    return Q


# 5. 使用模拟退火优化QUBO目标函数
initial_params = np.array([32, 64, 128])  # 初始超参数（卷积层输出通道数和全连接层特征数）

# 构建QUBO矩阵
Q = build_qubo_matrix(initial_params)

optimizer = kaiwu.classical.SimulatedAnnealingOptimizer(
    initial_temperature=200,  # 初始温度
    alpha=0.4,  # 降温系数
    cutoff_temperature=0.001,  # 截止温度
    iterations_per_t=10,  # 每个温度的迭代次数
    size_limit=1,  # 输出1个最优解
    flag_evolution_history=False,  # 不输出演化历史
    verbose=True  # 输出进度
)

# 传入QUBO矩阵进行优化
optimized_params = optimizer.solve(Q)
print(f"Optimized hyperparameters: {optimized_params}")

# 6. 映射QUBO解回实际的超参数值
# 假设QUBO解是一个二进制数组（例如 [1, 0, 1]），映射回整数
optimized_params = optimized_params.flatten()  # 将二维数组展平成一维数组

# 假设映射规则是：0映射为32，1映射为64，2映射为128（这只是一个示例）
param_map = {0: 32, 1: 64, 2: 128}

# 确保每个优化解都转换为合法的超参数（避免负值或不合理的值）
optimized_params = np.clip(optimized_params, 0, 2)  # 强制限定为 0, 1, 2

# 将二进制解映射到实际的超参数值
mapped_params = [param_map.get(int(x), 32) for x in optimized_params]

print(f"Mapped optimized hyperparameters: {mapped_params}")

# 7. 确保超参数是合理的
if len(mapped_params) == 3:
    model = CNN(*mapped_params)
else:
    print("Optimized parameters are not in the expected format")

# 8. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


# 9. 训练CNN模型
def train_model():
    model.train()
    for epoch in range(10):  # 训练10轮
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")


train_model()


# 10. 测试CNN模型
def test_model():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy on test set: {100 * correct / total}%")


test_model()


def qubo_model(parameters):
    """
    QUBO目标函数，用于计算当前超参数集的能量值
    参数：parameters - 当前超参数
    返回：目标函数值（能量值）
    """
    conv1_out_channels, conv2_out_channels, fc1_out_features = parameters

    # 约束1：确保 conv1_out_channels 在合理范围内 (例如 32, 64, 128)
    if conv1_out_channels not in [32, 64, 128]:
        penalty = 1000  # 惩罚项
    else:
        penalty = 0

    # 约束2：确保 conv2_out_channels 在合理范围内 (例如 32, 64, 128)
    if conv2_out_channels not in [32, 64, 128]:
        penalty += 1000

    # 约束3：确保 fc1_out_features 在合理范围内 (例如 128, 256)
    if fc1_out_features not in [128, 256]:
        penalty += 1000

    # 计算超参数的平方和，作为目标函数
    energy = np.sum(np.square(parameters)) + penalty
    return energy