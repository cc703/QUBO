import numpy as np
import pandas as pd
import kaiwu as kw

file_path_1 = ""
file_path_2 = ""
s = pd.read_csv(file_path_1, header=None).values.flatten()
w = pd.read_csv(file_path_2, header=None).values
n = s.shape[0]
k = 4
lambda_penalty = 10  # 惩罚系数
x = kw.qubo.ndarray((n,), "x", kw.qubo.Binary)

# 引入两个松弛变量
y1 = kw.qubo.Binary("y1")
y2 = kw.qubo.Binary("y2")
objective_linear = -0.5 * kw.qubo.quicksum([s[i] * x[i] for i in range(n)])
objective_quadratic = 0.5 * kw.qubo.quicksum([w[i, j] * x[i] * x[j]
                                               for i in range(n) for j in range(i+1, n)])
objective = objective_linear + objective_quadratic

# 创建 QUBO 模型并设置目标函数
qubo_model = kw.qubo.QuboModel()
qubo_model.set_objective(objective)

card_constraint = kw.qubo.quicksum([x[i] for i in range(n)]) + y1 + 2 * y2 - k
qubo_model.add_constraint(card_constraint == 0, "cardinality_constraint", penalty=lambda_penalty)
sol_dict, qubo_val = solver.solve_qubo(qubo_model)
#  精度求解
# 获取 QUBO 矩阵
Q = qubo_model.get_qubo_matrix(bit_width = 8)

