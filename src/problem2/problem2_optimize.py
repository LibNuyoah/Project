"""
=============================================================================
problem2_optimize.py — 问题二：基于NSGA-II的多目标充电桩布局优化
=============================================================================
功能：
  1. 建立三目标优化模型（成本、地理覆盖率、负荷均衡）
  2. 使用NSGA-II算法求解Pareto最优解集

输入文件：
  - results/tables/preprocess_data.npz
  - results/tables/表1_各区域供需缺口与建设紧迫度.xlsx

输出文件：
  - results/tables/Pareto前沿解集.xlsx
  - results/tables/NSGA-II收敛曲线数据.xlsx
  - results/tables/optimization_result.npz
=============================================================================
"""

import numpy as np
import pandas as pd
import os
import sys
import warnings
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from utils.paths import (
    FILE_Q2_TABLE1, FILE_Q2_PREPROCESS, FILE_Q2_PARETO,
    FILE_Q2_CONVERGENCE, FILE_Q2_OPTIMIZATION, RESULTS_TABLES
)

print('=' * 60)
print('问题二 NSGA-II 多目标优化求解（修正地理覆盖率模型）')
print('=' * 60)

# 加载预处理数据
data = np.load(FILE_Q2_PREPROCESS, allow_pickle=True)
spillover_matrix = data['spillover_matrix']
dist_matrix = data['dist_matrix']
service_radii = data['service_radii']

df_gap = pd.read_excel(FILE_Q2_TABLE1)

# =============================================================================
# 1. 全局参数
# =============================================================================
N_REGIONS = 10; N_VARS = 20
COST_FAST = 6.0; COST_SLOW = 0.8
CAP_FAST = 80; CAP_SLOW = 20
POWER_FAST = 120; POWER_SLOW = 7
SIMULTANEITY = 0.8; AVG_CHARGE_PER_CAR = 12.0
R_CORE = 1.5; R_NEW = 2.0; R_SUBURB = 2.5
COVERAGE_MIN = 0.90
ROBUST_DELTA = 1.0; RMSE = 208.52
POP_SIZE = 100; N_GENERATIONS = 500
P_CROSSOVER = 0.9; P_MUTATION = 0.1
ETA_C = 20; ETA_M = 20; MAX_NEW_PER_REGION = 300

# 附件1：现有桩数
existing_fast = np.array([129, 119, 99, 109, 76, 95, 45, 59, 39, 53])
existing_slow = np.array([86, 79, 66, 73, 50, 63, 30, 39, 26, 35])

# 面积数据
total_area = np.array([17.36, 14.25, 17.62, 110.07, 80.10, 60.08,
                       139.87, 120.04, 131.20, 22.30])
covered_area = np.array([14.02, 11.10, 14.50, 55.03, 32.44, 41.89,
                         35.02, 42.00, 26.17, 14.50])

current_coverage = covered_area / total_area

region_type_to_radius = {'老城核心区': R_CORE, '城市新区': R_NEW, '城郊/工业区': R_SUBURB}
region_radius = np.array([
    region_type_to_radius.get(
        df_gap.loc[df_gap['区域编号'] == i+1, '区域类型'].values[0], R_NEW
    ) for i in range(N_REGIONS)
])

pred_demand_kwh = df_gap['预测日均需求(MWh/日)'].values * 1000
pred_demand_trips = pred_demand_kwh / AVG_CHARGE_PER_CAR
pred_peak_kw = df_gap['峰值负荷(kW)'].values
grid_capacity_kw = df_gap['电网总容量(kW)'].values

print(f'\n当前地理覆盖率: {np.round(current_coverage * 100, 1)}')
print(f'覆盖率<90%的区域: {np.sum(current_coverage < 0.9)} 个')
print(f'各区域服务半径(km): {region_radius}')

SINGLE_COVERAGE_AREA = np.pi * region_radius**2


def compute_coverage(delta_fast, delta_slow):
    pop_size = delta_fast.shape[0]
    new_coverage = np.zeros((pop_size, N_REGIONS))
    delta_total = 2.0 * delta_fast + 1.0 * delta_slow
    for i in range(N_REGIONS):
        uncovered_ratio = 1.0 - current_coverage[i]
        marginal_per_charger = SINGLE_COVERAGE_AREA[i] * (uncovered_ratio ** 1.2)
        added_area = delta_total[:, i] * marginal_per_charger
        spillover_contribution = 0.0
        for j in range(N_REGIONS):
            if i != j and spillover_matrix[j, i] > 0.05:
                spillover_contribution += (spillover_matrix[j, i] *
                                           delta_total[:, j] *
                                           SINGLE_COVERAGE_AREA[j] * 0.1)
        new_covered = covered_area[i] + added_area + spillover_contribution
        new_coverage[:, i] = np.minimum(new_covered / total_area[i], 1.0)
    return new_coverage


def compute_coverage_standalone(delta_fast, delta_slow):
    pop_size = delta_fast.shape[0]
    cov = np.zeros((pop_size, N_REGIONS))
    delta_total = 2.0 * delta_fast + 1.0 * delta_slow
    for i in range(N_REGIONS):
        marginal = SINGLE_COVERAGE_AREA[i] * ((1.0 - current_coverage[i]) ** 1.2)
        added = delta_total[:, i] * marginal
        cov[:, i] = np.minimum((covered_area[i] + added) / total_area[i], 1.0)
    return cov


def compute_service_capacity(delta_fast, delta_slow):
    pop_size = delta_fast.shape[0]
    capacity = np.zeros((pop_size, N_REGIONS))
    for i in range(N_REGIONS):
        cap_i = (CAP_FAST * (existing_fast[i] + delta_fast[:, i]) +
                 CAP_SLOW * (existing_slow[i] + delta_slow[:, i]))
        for j in range(N_REGIONS):
            if spillover_matrix[j, i] > 0.05:
                cap_j = (CAP_FAST * (existing_fast[j] + delta_fast[:, j]) +
                         CAP_SLOW * (existing_slow[j] + delta_slow[:, j]))
                capacity[:, i] += spillover_matrix[j, i] * cap_j * 0.05
        capacity[:, i] += cap_i
    return capacity


def decode_variables(x):
    x = np.atleast_2d(x)
    delta_fast = np.clip(np.round(x[:, 0::2]), 0, MAX_NEW_PER_REGION)
    delta_slow = np.clip(np.round(x[:, 1::2]), 0, MAX_NEW_PER_REGION)
    delta_total = delta_fast + delta_slow
    for i in range(N_REGIONS):
        mask = delta_total[:, i] > MAX_NEW_PER_REGION
        if mask.any():
            scale = MAX_NEW_PER_REGION / delta_total[mask, i]
            delta_fast[mask, i] = np.floor(delta_fast[mask, i] * scale)
            delta_slow[mask, i] = np.floor(delta_slow[mask, i] * scale)
    return delta_fast, delta_slow


def objective_cost(delta_fast, delta_slow):
    return COST_FAST * np.sum(delta_fast, axis=1) + COST_SLOW * np.sum(delta_slow, axis=1)


def objective_coverage(delta_fast, delta_slow):
    cov = compute_coverage(delta_fast, delta_slow)
    return np.mean(cov, axis=1)


def objective_balance(delta_fast, delta_slow):
    pop_size = delta_fast.shape[0]
    load_rates = np.zeros((pop_size, N_REGIONS))
    delta_load = SIMULTANEITY * (POWER_FAST * delta_fast + POWER_SLOW * delta_slow)
    for i in range(N_REGIONS):
        load_rates[:, i] = (pred_peak_kw[i] + delta_load[:, i]) / grid_capacity_kw[i]
    return np.var(load_rates, axis=1)


def check_constraints(delta_fast, delta_slow):
    pop_size = delta_fast.shape[0]
    feasible = np.ones(pop_size, dtype=bool)
    cov_standalone = compute_coverage_standalone(delta_fast, delta_slow)
    for i in range(N_REGIONS):
        feasible = feasible & (cov_standalone[:, i] >= COVERAGE_MIN)
    capacity = compute_service_capacity(delta_fast, delta_slow)
    trips_robust = pred_demand_trips + ROBUST_DELTA * RMSE / AVG_CHARGE_PER_CAR
    for i in range(N_REGIONS):
        feasible = feasible & (capacity[:, i] >= trips_robust[i])
    delta_load = SIMULTANEITY * (POWER_FAST * delta_fast + POWER_SLOW * delta_slow)
    for i in range(N_REGIONS):
        feasible = feasible & (pred_peak_kw[i] + delta_load[:, i] <= grid_capacity_kw[i])
    return feasible


def evaluate_population(x):
    delta_fast, delta_slow = decode_variables(x)
    obj1 = objective_cost(delta_fast, delta_slow)
    obj2 = objective_coverage(delta_fast, delta_slow)
    obj3 = objective_balance(delta_fast, delta_slow)
    feasible = check_constraints(delta_fast, delta_slow)
    return obj1, obj2, obj3, feasible


def initialize_population():
    pop = np.zeros((POP_SIZE, N_VARS))
    urgency = df_gap.sort_values('区域编号')['建设紧迫度指数(0-100)'].values
    urgency_norm = urgency / urgency.max()
    for i in range(N_REGIONS):
        base_fast = int(urgency_norm[i] * 40)
        base_slow = int(urgency_norm[i] * 80)
        pop[:, 2*i] = np.random.randint(0, max(base_fast + 10, 20), POP_SIZE)
        pop[:, 2*i+1] = np.random.randint(0, max(base_slow + 10, 40), POP_SIZE)
    return pop


def non_dominated_sort(obj1, obj2, obj3, feasible):
    pop_size = len(obj1)
    dominated_count = np.zeros(pop_size, dtype=int)
    dominates_list = [[] for _ in range(pop_size)]
    for p in range(pop_size):
        for q in range(p + 1, pop_size):
            if feasible[p] and not feasible[q]:
                dominates_list[p].append(q); dominated_count[q] += 1
            elif not feasible[p] and feasible[q]:
                dominated_count[p] += 1; dominates_list[q].append(p)
            elif feasible[p] and feasible[q]:
                p_better = (obj1[p] <= obj1[q] and obj2[p] >= obj2[q] and obj3[p] <= obj3[q])
                p_strict = (obj1[p] < obj1[q] or obj2[p] > obj2[q] or obj3[p] < obj3[q])
                q_better = (obj1[q] <= obj1[p] and obj2[q] >= obj2[p] and obj3[q] <= obj3[p])
                q_strict = (obj1[q] < obj1[p] or obj2[q] > obj2[p] or obj3[q] < obj3[p])
                if p_better and p_strict:
                    dominates_list[p].append(q); dominated_count[q] += 1
                elif q_better and q_strict:
                    dominated_count[p] += 1; dominates_list[q].append(p)
    fronts = []
    current_front = [i for i in range(pop_size) if dominated_count[i] == 0]
    while current_front:
        fronts.append(current_front)
        next_front = []
        for p in current_front:
            for q in dominates_list[p]:
                dominated_count[q] -= 1
                if dominated_count[q] == 0:
                    next_front.append(q)
        current_front = next_front
    return fronts


def crowding_distance(obj1, obj2, obj3, front_indices):
    n = len(front_indices)
    if n <= 2:
        return np.full(n, np.inf)
    distance = np.zeros(n)
    for obj in [obj1, obj2, obj3]:
        values = -obj[front_indices] if obj is obj2 else obj[front_indices]
        sorted_idx = np.argsort(values)
        values_sorted = values[sorted_idx]
        distance[sorted_idx[0]] = np.inf; distance[sorted_idx[-1]] = np.inf
        value_range = values_sorted[-1] - values_sorted[0]
        if value_range > 0:
            for i in range(1, n - 1):
                distance[sorted_idx[i]] += (values_sorted[i + 1] - values_sorted[i - 1]) / value_range
    return distance


def binary_tournament_selection(pop, obj1, obj2, obj3, feasible, fronts, crowding):
    pop_size = len(pop)
    selected = np.zeros((pop_size, N_VARS))
    rank = np.full(pop_size, 999, dtype=int)
    for fi, front in enumerate(fronts):
        for idx in front:
            rank[idx] = fi
    for i in range(pop_size):
        a, b = np.random.choice(pop_size, 2, replace=False)
        if rank[a] < rank[b]: winner = a
        elif rank[b] < rank[a]: winner = b
        elif crowding[a] > crowding[b]: winner = a
        else: winner = b
        selected[i] = pop[winner]
    return selected


def sbx_crossover(p1, p2):
    if np.random.random() > P_CROSSOVER:
        return p1.copy(), p2.copy()
    c1, c2 = p1.copy(), p2.copy()
    for i in range(N_VARS):
        if np.random.random() < 0.5:
            u = np.random.random()
            beta = (2*u)**(1/(ETA_C+1)) if u <= 0.5 else (1/(2*(1-u)))**(1/(ETA_C+1))
            c1[i] = np.clip(np.round(0.5*((1+beta)*p1[i] + (1-beta)*p2[i])), 0, MAX_NEW_PER_REGION)
            c2[i] = np.clip(np.round(0.5*((1-beta)*p1[i] + (1+beta)*p2[i])), 0, MAX_NEW_PER_REGION)
    return c1, c2


def polynomial_mutation(ind):
    mutant = ind.copy()
    for i in range(N_VARS):
        if np.random.random() < P_MUTATION:
            u = np.random.random()
            delta = (2*u)**(1/(ETA_M+1)) - 1 if u < 0.5 else 1 - (2*(1-u))**(1/(ETA_M+1))
            mutant[i] = np.clip(np.round(ind[i] + delta * MAX_NEW_PER_REGION), 0, MAX_NEW_PER_REGION)
    return mutant


# =============================================================================
# NSGA-II 主循环
# =============================================================================
print('\n' + '=' * 60)
print('开始 NSGA-II 优化（修正覆盖率模型）')
print('=' * 60)

np.random.seed(42)
pop = initialize_population()
obj1, obj2, obj3, feasible = evaluate_population(pop)

convergence_history = {
    'generation': [], 'min_cost': [], 'max_coverage': [],
    'min_variance': [], 'n_feasible': [], 'n_pareto_front1': [],
}

for gen in range(N_GENERATIONS):
    fronts = non_dominated_sort(obj1, obj2, obj3, feasible)
    crowding = np.zeros(POP_SIZE)
    for front in fronts:
        crowding[front] = crowding_distance(obj1, obj2, obj3, front)

    feasible_idx = np.where(feasible)[0]
    conv = convergence_history
    conv['generation'].append(gen)
    if len(feasible_idx) > 0:
        conv['min_cost'].append(np.min(obj1[feasible_idx]))
        conv['max_coverage'].append(np.max(obj2[feasible_idx]))
        conv['min_variance'].append(np.min(obj3[feasible_idx]))
    else:
        conv['min_cost'].append(np.inf); conv['max_coverage'].append(0); conv['min_variance'].append(np.inf)
    conv['n_feasible'].append(np.sum(feasible))
    conv['n_pareto_front1'].append(len(fronts[0]) if fronts else 0)

    if gen % 50 == 0:
        fc = np.sum(feasible); mc = conv['min_cost'][-1]; cv = conv['max_coverage'][-1]
        print(f'Gen {gen:3d} | 可行:{fc:3d} | F1:{len(fronts[0]):2d} | '
              f'MinCost:{"inf" if np.isinf(mc) else f"{mc:.0f}万":>8s} | '
              f'MaxCov:{cv:.4f} | MinVar:{conv["min_variance"][-1]:.6f}')

    selected = binary_tournament_selection(pop, obj1, obj2, obj3, feasible, fronts, crowding)
    offspring = np.zeros_like(selected)
    for i in range(0, POP_SIZE, 2):
        c1, c2 = sbx_crossover(selected[i], selected[min(i+1, POP_SIZE-1)])
        offspring[i] = polynomial_mutation(c1)
        if i + 1 < POP_SIZE:
            offspring[i+1] = polynomial_mutation(c2)
    o1, o2, o3, of = evaluate_population(offspring)
    cp = np.vstack([pop, offspring])
    co1 = np.hstack([obj1, o1]); co2 = np.hstack([obj2, o2]); co3 = np.hstack([obj3, o3])
    cof = np.hstack([feasible, of])
    cfronts = non_dominated_sort(co1, co2, co3, cof)
    ccrowd = np.zeros(2*POP_SIZE)
    for fr in cfronts:
        ccrowd[fr] = crowding_distance(co1, co2, co3, fr)

    new_pop = np.zeros((POP_SIZE, N_VARS))
    no1 = np.zeros(POP_SIZE); no2 = np.zeros(POP_SIZE); no3 = np.zeros(POP_SIZE)
    nof = np.zeros(POP_SIZE, dtype=bool)
    cnt = 0
    for fr in cfronts:
        if cnt + len(fr) <= POP_SIZE:
            for idx in fr:
                new_pop[cnt] = cp[idx]; no1[cnt] = co1[idx]; no2[cnt] = co2[idx]
                no3[cnt] = co3[idx]; nof[cnt] = cof[idx]; cnt += 1
        else:
            rem = POP_SIZE - cnt
            fcrd = ccrowd[fr]
            for idx in np.argsort(fcrd)[::-1][:rem]:
                new_pop[cnt] = cp[fr[idx]]; no1[cnt] = co1[fr[idx]]
                no2[cnt] = co2[fr[idx]]; no3[cnt] = co3[fr[idx]]; nof[cnt] = cof[fr[idx]]
                cnt += 1
            break
    pop, obj1, obj2, obj3, feasible = new_pop, no1, no2, no3, nof

print(f'\nNSGA-II完成！可行解: {np.sum(feasible)}, F1规模: {convergence_history["n_pareto_front1"][-1]}')

# =============================================================================
# 提取Pareto前沿
# =============================================================================
print('\n提取Pareto前沿...')
final_fronts = non_dominated_sort(obj1, obj2, obj3, feasible)
pareto_indices = [i for i in final_fronts[0] if feasible[i]]

if len(pareto_indices) < 5:
    print(f'可行Pareto解不足({len(pareto_indices)}个)，放宽约束...')
    cov_all = compute_coverage(*decode_variables(pop))
    pareto_indices = [i for i in final_fronts[0] if np.all(cov_all[i] >= 0.80)]

print(f'Pareto解数量: {len(pareto_indices)}')

pareto_pop = pop[pareto_indices]
pareto_obj1 = obj1[pareto_indices]; pareto_obj2 = obj2[pareto_indices]; pareto_obj3 = obj3[pareto_indices]
pareto_fast, pareto_slow = decode_variables(pareto_pop)

print(f'成本范围: {pareto_obj1.min():.0f} ~ {pareto_obj1.max():.0f} 万元')
print(f'覆盖率范围: {pareto_obj2.min():.4f} ~ {pareto_obj2.max():.4f}')
print(f'方差范围: {pareto_obj3.min():.6f} ~ {pareto_obj3.max():.6f}')

# =============================================================================
# 保存结果
# =============================================================================
pareto_records = []
for k in range(len(pareto_indices)):
    rec = {'解编号': k+1, '总成本_万元': pareto_obj1[k],
           '平均覆盖率': pareto_obj2[k], '负荷率方差': pareto_obj3[k]}
    for i in range(N_REGIONS):
        rec[f'区域{i+1}_新增快充'] = int(pareto_fast[k, i])
        rec[f'区域{i+1}_新增慢充'] = int(pareto_slow[k, i])
    pareto_records.append(rec)

pd.DataFrame(pareto_records).to_excel(FILE_Q2_PARETO, index=False)
pd.DataFrame(convergence_history).to_excel(FILE_Q2_CONVERGENCE, index=False)

np.savez(FILE_Q2_OPTIMIZATION,
         pareto_pop=pareto_pop, pareto_obj1=pareto_obj1,
         pareto_obj2=pareto_obj2, pareto_obj3=pareto_obj3,
         pareto_fast=pareto_fast, pareto_slow=pareto_slow,
         convergence_history=convergence_history)

print('优化结果已保存。')
print('=' * 60)
print('problem2_optimize.py 运行完成！')
print('=' * 60)
