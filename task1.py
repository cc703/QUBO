import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.impute import SimpleImputer
import kaiwu as kw
import matplotlib.pyplot as plt
import seaborn as sns

# 文件中并未存在列名，自定义
column = [
    'checking_account', 'duration', 'credit_history', 'purpose', 'credit_amount', 'savings', 'employment',
    'personal_status', 'other_parties', 'residence_since', 'property_magnitude', 'other_payment_plans',
    'housing', 'existing_credits', 'job', 'num_people_liable', 'own_telephone', 'foreign_worker', 'class'
]

# 加载数据集
data = pd.read_csv('statlog+german+credit+data/german.data-numeric',
                   header=None,
                   delimiter=' ',
                   names=column)

# 计算特征量的缺失值比例,转换为百分比
missing_data_ratio = data.isnull().mean() * 100

# 编码分类转化为数值变量
encoder = LabelEncoder()

# 需要编码的列
cate_columns = ['checking_account', 'credit_history', 'purpose', 'savings', 'employment',
                'personal_status', 'other_parties', 'property_magnitude', 'other_payment_plans',
                'housing', 'job', 'own_telephone', 'foreign_worker']

# 对每一列进行编码
for col in cate_columns:
    data[col] = encoder.fit_transform(data[col])

# 处理缺失值
# 使用 SimpleImputer 填充缺失值----使用均值填充
imputer_X = SimpleImputer(strategy='mean')
# 目标变量的插补---使用众数填充---以去大众化代替
imputer_y = SimpleImputer(strategy='most_frequent')

# 对整个数据集进行插补处理
X = data.drop(columns=['class'])  # 特征
y = data['class']  # 目标变量

# 填充X、Y
X_imputed = imputer_X.fit_transform(X)
y_imputed = imputer_y.fit_transform(y.values.reshape(-1, 1)).ravel()

# 计算每个特征的准确率、F1分数和AUC
accuracy_scores = []
f1_scores = []
auc_scores = []
for i, col in enumerate(data.columns[:-1]):  # 排除class列
    X_single_feature = X_imputed[:, i].reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X_single_feature, y_imputed, test_size=0.3, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # 计算准确率、F1分数和AUC
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    accuracy_scores.append(accuracy)
    f1_scores.append(f1)
    auc_scores.append(auc)

# 权重和惩罚系数的定义
alpha = 0.5  # 预测准确率的权重
beta = 0.5   # 缺失数据比例的权重
lambda_penalty = 100  # 惩罚系数
constraint_penalty = 1000  # 确保只有一个特征被选中的惩罚系数
missing_data_penalty = 1000  # 处理缺失数据比例高于50的惩罚系数
auc_penalty = 1000  # 处理AUC低于0.7的惩罚系数
max_selected_features = 5  # 设置最多选择5个特征
complexity_penalty = 1000  # 复杂度惩罚系数

# 构建目标函数的QUBO系数
qubo_coefficients = {}

# 设置主对角线系数（特征选择准确率和缺失数据的加权值）
for i in range(len(data.columns) - 1):  # 排除class列
    qubo_coefficients[(i, i)] = alpha * accuracy_scores[i] + beta * missing_data_ratio[i]

# 添加缺失值约束：如果缺失比例大于50，则增加惩罚
for i in range(len(data.columns) - 1):  # 排除class列
    if missing_data_ratio[i] > 60:
        qubo_coefficients[(i, i)] += missing_data_penalty  # 为缺失比例大于50的特征增加惩罚

# 添加AUC约束：如果AUC小于0.7，则增加惩罚
for i in range(len(data.columns) - 1):  # 排除class列
    if auc_scores[i] < 0.8:
        qubo_coefficients[(i, i)] += auc_penalty  # 为AUC小于0.7的特征增加惩罚



# 构建QUBO矩阵（将QUBO系数字典转换为矩阵）
num_features = len(data.columns) - 1  # 排除class列
qubo_matrix = np.zeros((num_features, num_features))

# 将系数填充到QUBO矩阵中
for (i, j), value in qubo_coefficients.items():
    qubo_matrix[i, j] = value
    qubo_matrix[j, i] = value  # 确保对称性

# 现在可以将这个QUBO矩阵传递给模拟退火优化器
optimizer = kw.classical.SimulatedAnnealingOptimizer(
    initial_temperature=200,       # 初始温度
    alpha=0.4,                    # 降温系数
    cutoff_temperature=0.001,      # 截止温度
    iterations_per_t=10,           # 每个温度的迭代次数
    size_limit=1,                  # 输出1个最优解
    flag_evolution_history=False,  # 不输出演化历史
    verbose=True                   # 输出进度
)

# 执行模拟退火优化
solution = optimizer.solve(qubo_matrix)

# 打印最优解
print("\n最优解:")
print(solution)

solution = np.array(solution).flatten()

# 映射二进制解回特征名称
selected_features = [data.columns[i] for i in range(len(solution)) if solution[i] == 1]

# 输出最优选择的特征
print("\n最优特征子集：")
print(selected_features)

# 打印每个特征的准确率、F1分数和AUC
for feature in selected_features:
    idx = data.columns.get_loc(feature)
    print(f"特征: {feature}")
    print(f"  准确率: {accuracy_scores[idx]:.5f}")
    print(f"  F1分数: {f1_scores[idx]:.5f}")
    print(f"  AUC: {auc_scores[idx]:.5f}\n")

# 可视化特征的准确率、F1分数、AUC

# 设置绘图风格
sns.set(style="whitegrid")

# 创建一个包含准确率、F1分数、AUC的DataFrame
feature_performance = pd.DataFrame({
    'Feature': data.columns[:-1],
    'Accuracy': accuracy_scores,
    'F1 Score': f1_scores,
    'AUC': auc_scores
})

# 绘制准确率、F1分数和AUC的柱状图
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 绘制准确率
sns.barplot(x='Accuracy', y='Feature', data=feature_performance.sort_values('Accuracy', ascending=False), ax=axes[0])
axes[0].set_title('Feature Accuracy')

# 绘制F1分数
sns.barplot(x='F1 Score', y='Feature', data=feature_performance.sort_values('F1 Score', ascending=False), ax=axes[1])
axes[1].set_title('Feature F1 Score')

# 绘制AUC
sns.barplot(x='AUC', y='Feature', data=feature_performance.sort_values('AUC', ascending=False), ax=axes[2])
axes[2].set_title('Feature AUC')


# 调整子图布局，避免重叠
plt.tight_layout()

# 显示图表
plt.show()
