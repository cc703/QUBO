import numpy as np
import pandas as pd
import kaiwu as kw

s_vector = pd.read_csv("s_vector.csv", header=None).values.flatten()
w_matrix = pd.read_csv("w_weight.csv", header=None).values
n = s_vector.shape[0]
k = 4
lambda_penalty = 10  # 惩罚系数
x = kw.qubo.ndarray((n,), "x", kw.qubo.Binary)

# 引入两个松弛变量
y1 = kw.qubo.Binary("y1")
y2 = kw.qubo.Binary("y2")
objective_linear = -0.5 * kw.qubo.quicksum([s_vector[i] * x[i] for i in range(n)])
objective_quadratic = 0.5 * kw.qubo.quicksum([w_matrix[i, j] * x[i] * x[j]
                                               for i in range(n) for j in range(i+1, n)])
objective = objective_linear + objective_quadratic

# 创建 QUBO 模型并设置目标函数
qubo_model = kw.qubo.QuboModel()
qubo_model.set_objective(objective)

card_constraint = kw.qubo.quicksum([x[i] for i in range(n)]) + y1 + 2 * y2 - k
qubo_model.add_constraint(card_constraint == 0, "cardinality_constraint", penalty=lambda_penalty)

# 求解 QUBO 模型
solver = kw.solver.SimpleSolver(
    kw.classical.SimulatedAnnealingOptimizer(initial_temperature=100,
                                               alpha=0.99,
                                               cutoff_temperature=0.001,
                                               iterations_per_t=10,
                                               size_limit=100))

sol_dict, qubo_val = solver.solve_qubo(qubo_model)

unsatisfied_count, res_dict = qubo_model.verify_constraint(sol_dict)
print("Unsatisfied constraints: ", unsatisfied_count)
print("Constraint energy components:", res_dict)
print("QUBO objective value (energy): {:.4f}".format(kw.qubo.get_val(qubo_model.objective, sol_dict)))

# 获取 QUBO 矩阵
Q = qubo_model.get_qubo_matrix(bit_width = 8)
#print(Q)
pd.DataFrame(Q).to_csv("bit8.csv", index=False, header=False)

