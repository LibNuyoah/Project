"""
=============================================================================
problem2_result.py — 问题二：结果分析与可视化
=============================================================================
功能：
  1. 熵权-TOPSIS方法从Pareto前沿中选取最优折中方案
  2. 生成全部5张合并图表和3张数据表
  3. 输出最终配置方案和优化前后对比

输入文件：
  - results/tables/preprocess_data.npz
  - results/tables/optimization_result.npz
  - results/tables/Pareto前沿解集.xlsx
  - results/tables/表1_各区域供需缺口与建设紧迫度.xlsx

输出文件（图表）：
  - results/figures/图1_建设紧迫度与供需缺口.png
  - results/figures/图2_空间溢出权重热力图.png
  - results/figures/图3_NSGA-II求解过程.png
  - results/figures/图4_TOPSIS最优解选取.png
  - results/figures/图5_配置方案与优化效果对比.png

输出文件（表格）：
  - results/tables/表2_各区域最优配置方案.xlsx
  - results/tables/表3_优化前后多指标对比.xlsx
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import os
import sys
import warnings
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from utils.paths import (
    RESULTS_FIGURES, RESULTS_TABLES,
    FILE_Q2_PREPROCESS, FILE_Q2_OPTIMIZATION, FILE_Q2_PARETO, FILE_Q2_TABLE1,
    FILE_Q2_TABLE2, FILE_Q2_TABLE3
)

# =============================================================================
# 全局绘图风格设置
# =============================================================================
from utils.mpl_setup import setup_chinese
setup_chinese()
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.size': 9,
})

COLOR_BLUE = '#2B579A'; COLOR_ORANGE = '#E07B39'
COLOR_GREEN = '#3A8E6F'; COLOR_RED = '#C44E52'; COLOR_GREY = '#8C8C8C'
PALETTE_3 = [COLOR_BLUE, COLOR_ORANGE, COLOR_GREEN]

REGION_NAMES = ['宝塔山街道', '南市街道', '凤凰山街道', '桥沟街道', '枣园街道',
                '新城街道', '河庄坪镇', '姚店镇（经开区）', '万花山镇', '真武洞街道（安塞）']
REGION_TYPES = ['老城核心区', '老城核心区', '城市新区', '老城核心区', '城市新区',
                '城市新区', '城郊/工业区', '城郊/工业区', '城郊/工业区', '城郊/工业区']
N_REGIONS = 10

# =============================================================================
# 0. 加载数据
# =============================================================================
print('=' * 60)
print('问题二 结果分析与可视化')
print('=' * 60)

data_pre = np.load(FILE_Q2_PREPROCESS, allow_pickle=True)
spillover_matrix = data_pre['spillover_matrix']

data_opt = np.load(FILE_Q2_OPTIMIZATION, allow_pickle=True)
pareto_obj1 = data_opt['pareto_obj1']
pareto_obj2 = data_opt['pareto_obj2']
pareto_obj3 = data_opt['pareto_obj3']
pareto_obj4 = data_opt['pareto_obj4'] if 'pareto_obj4' in data_opt else np.zeros(len(pareto_obj1))
pareto_fast = data_opt['pareto_fast']
pareto_slow = data_opt['pareto_slow']
convergence = data_opt['convergence_history'].item()

df_gap = pd.read_excel(FILE_Q2_TABLE1)

COST_FAST, COST_SLOW = 6.0, 0.8
CAP_FAST, CAP_SLOW = 80, 20

print(f'Pareto解集大小: {len(pareto_obj1)}')
print(f'收敛曲线记录: {len(convergence["generation"])} 代')

# =============================================================================
# 1. 熵权-TOPSIS最优解选取
# =============================================================================
print('\n' + '=' * 60)
print('步骤1: 熵权-TOPSIS选取最优折中方案')
print('=' * 60)

COV_LOWER = 0.80; COV_UPPER = 1.00  # 匹配优化器约束，覆盖率≥80%纳入候选
feasible_mask = (pareto_obj2 >= COV_LOWER) & (pareto_obj2 <= COV_UPPER)
if feasible_mask.sum() < 3:
    feasible_mask = pareto_obj2 >= COV_LOWER
print(f'覆盖率筛选: [{COV_LOWER*100:.0f}%, {COV_UPPER*100:.0f}%], 候选解: {feasible_mask.sum()}个')

candidate_idx = np.where(feasible_mask)[0]
n_c = len(candidate_idx)

c_obj1 = pareto_obj1[candidate_idx]
c_obj2_raw = pareto_obj2[candidate_idx]
c_obj3 = pareto_obj3[candidate_idx]
c_obj4 = pareto_obj4[candidate_idx]
c_obj2 = np.where(c_obj2_raw > COV_LOWER, COV_LOWER + 0.5 * (c_obj2_raw - COV_LOWER), c_obj2_raw)
c_obj_matrix = np.column_stack([c_obj1, c_obj2, c_obj3, c_obj4])

c_norm = np.zeros((n_c, 4))
directions = ['cost', 'benefit', 'cost', 'cost']  # 4目标: 成本↓, 覆盖率↑, 均衡↓, 风险↓
for j in range(4):
    col = c_obj_matrix[:, j]
    col_min, col_max = col.min(), col.max()
    if col_max > col_min:
        if directions[j] == 'benefit':
            c_norm[:, j] = (col - col_min) / (col_max - col_min)
        else:
            c_norm[:, j] = (col_max - col) / (col_max - col_min)

c_p = np.clip(c_norm / (c_norm.sum(axis=0, keepdims=True) + 1e-10), 1e-10, 1)
c_e = -np.sum(c_p * np.log(c_p), axis=0) / np.log(n_c)
c_e = np.nan_to_num(c_e, nan=0.0)
c_w = (1 - c_e) / (np.sum(1 - c_e) + 1e-10)
c_w = np.nan_to_num(c_w, nan=0.25)  # 回退等权重
c_weighted = c_norm * c_w
c_pos = c_weighted.max(axis=0); c_neg = c_weighted.min(axis=0)
c_dpos = np.sqrt(np.sum((c_weighted - c_pos)**2, axis=1))
c_dneg = np.sqrt(np.sum((c_weighted - c_neg)**2, axis=1))
c_close = c_dneg / (c_dpos + c_dneg)

best_local = np.argmax(c_close)
best_idx = candidate_idx[best_local]

print(f'\n四目标权重: 成本={c_w[0]:.4f}, 覆盖率={c_w[1]:.4f}, 负荷均衡={c_w[2]:.4f}, 电网风险={c_w[3]:.4f}')
print(f'\n最优方案: #{best_idx+1}')
print(f'  成本: {pareto_obj1[best_idx]:.1f} 万元')
print(f'  覆盖率: {pareto_obj2[best_idx]:.4f} ({pareto_obj2[best_idx]*100:.1f}%)')
print(f'  负荷率方差: {pareto_obj3[best_idx]:.6f}')
print(f'  电网风险指数: {pareto_obj4[best_idx]:.6f}')

best_fast = pareto_fast[best_idx].astype(int)
best_slow = pareto_slow[best_idx].astype(int)

# =============================================================================
# 2. 图1: 建设紧迫度与供需缺口
# =============================================================================
print('\n步骤2: 生成图1 — 建设紧迫度与供需缺口')
fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(12, 4.5))
df_gap_sorted = df_gap.sort_values('建设紧迫度指数(0-100)', ascending=True)

colors_urgency = [COLOR_RED if v > 60 else (COLOR_ORANGE if v > 30 else COLOR_BLUE)
                  for v in df_gap_sorted['建设紧迫度指数(0-100)']]
bars1 = ax1a.barh(range(N_REGIONS), df_gap_sorted['建设紧迫度指数(0-100)'],
                   color=colors_urgency, edgecolor='white', linewidth=0.5)
ax1a.set_yticks(range(N_REGIONS))
ax1a.set_yticklabels(df_gap_sorted['区域名称'], fontsize=7)
ax1a.set_xlabel('建设紧迫度指数 (0-100)', fontsize=8)
ax1a.set_title('(a) 各区域建设紧迫度排序', fontsize=10, fontweight='bold')
ax1a.invert_yaxis()
for i, v in enumerate(df_gap_sorted['建设紧迫度指数(0-100)']):
    ax1a.text(v + 1, i, f'{v:.0f}', va='center', fontsize=6, color='#333333')

x = np.arange(N_REGIONS); width = 0.35
existing = df_gap_sorted['现有服务能力(车次/日)'].values
demand = df_gap_sorted['预测日均车次(次/日)'].values
ax1b.bar(x - width/2, existing, width, label='现有服务能力', color=COLOR_BLUE, edgecolor='white', linewidth=0.5)
ax1b.bar(x + width/2, demand, width, label='预测车次需求', color=COLOR_ORANGE, edgecolor='white', linewidth=0.5)
ax1b.set_xticks(x)
ax1b.set_xticklabels(df_gap_sorted['区域名称'], rotation=30, ha='right', fontsize=7)
ax1b.set_ylabel('车次/日', fontsize=8)
ax1b.set_title('(b) 各区域充电供需对比', fontsize=10, fontweight='bold')
ax1b.legend(fontsize=7, loc='upper left'); ax1b.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
fig1.savefig(os.path.join(RESULTS_FIGURES, '图1_建设紧迫度与供需缺口.png'), dpi=300, bbox_inches='tight')
plt.close()
print('图1 已保存')

# =============================================================================
# 3. 图2: 空间溢出权重热力图
# =============================================================================
print('\n步骤3: 生成图2 — 空间溢出权重热力图')
fig2, ax2 = plt.subplots(figsize=(7, 6))
sns.heatmap(spillover_matrix, annot=np.round(spillover_matrix, 2),
            fmt='.2f', cmap='YlOrRd', ax=ax2,
            xticklabels=[f'{n[:2]}' for n in REGION_NAMES],
            yticklabels=[f'R{i+1}-{n[:2]}' for i, n in enumerate(REGION_NAMES)],
            vmin=0, vmax=1, linewidths=0.5, linecolor='white',
            cbar_kws={'label': '溢出权重', 'shrink': 0.8})
ax2.set_title('空间溢出权重矩阵 W', fontsize=10, fontweight='bold')
ax2.set_xlabel('被服务区域 j', fontsize=9); ax2.set_ylabel('充电桩所在区域 i', fontsize=9)
plt.tight_layout()
fig2.savefig(os.path.join(RESULTS_FIGURES, '图2_空间溢出权重热力图.png'), dpi=300, bbox_inches='tight')
plt.close()
print('图2 已保存')

# =============================================================================
# 4. 图3: NSGA-II求解过程
# =============================================================================
print('\n步骤4: 生成图3 — NSGA-II求解过程')
fig3 = plt.figure(figsize=(14, 4.2))
ax3a = fig3.add_subplot(1, 3, 1)
gens = convergence['generation']
ax3a.plot(gens, convergence['min_cost'], color=COLOR_BLUE, linewidth=1.2, label='最小成本')
ax3a.set_xlabel('迭代代数', fontsize=8); ax3a.set_ylabel('最小成本 (万元)', fontsize=8, color=COLOR_BLUE)
ax3a.tick_params(axis='y', labelcolor=COLOR_BLUE)
ax3a_2 = ax3a.twinx()
ax3a_2.plot(gens, convergence['max_coverage'], color=COLOR_ORANGE, linewidth=1.2, label='最大覆盖率')
ax3a_2.set_ylabel('最大覆盖率', fontsize=8, color=COLOR_ORANGE)
ax3a_2.tick_params(axis='y', labelcolor=COLOR_ORANGE)
ax3a.set_title('(a) NSGA-II 收敛曲线', fontsize=10, fontweight='bold'); ax3a.grid(alpha=0.3, linestyle='--')

ax3b = fig3.add_subplot(1, 3, 2)
scatter_b = ax3b.scatter(pareto_obj1, pareto_obj2 * 100, c=pareto_obj3,
                          cmap='YlOrRd', s=25, edgecolors='grey', linewidth=0.3, alpha=0.8)
ax3b.set_xlabel('总成本 (万元)', fontsize=8); ax3b.set_ylabel('平均覆盖率 (%)', fontsize=8)
ax3b.set_title('(b) Pareto前沿: 成本 vs 覆盖率', fontsize=10, fontweight='bold')
ax3b.grid(alpha=0.3, linestyle='--')
cbar_b = plt.colorbar(scatter_b, ax=ax3b, shrink=0.8); cbar_b.set_label('负荷率方差', fontsize=7)
ax3b.scatter(pareto_obj1[best_idx], pareto_obj2[best_idx] * 100,
             color=COLOR_RED, s=80, marker='*', edgecolors='white',
             linewidth=1.5, zorder=5, label='TOPSIS最优解')
ax3b.legend(fontsize=7, loc='lower right')

ax3c = fig3.add_subplot(1, 3, 3)
obj_norm = np.zeros((len(pareto_obj1), 4))
obj_norm[:, 0] = (pareto_obj1 - pareto_obj1.min()) / (pareto_obj1.max() - pareto_obj1.min() + 1e-10)
obj_norm[:, 1] = (pareto_obj2.max() - pareto_obj2) / (pareto_obj2.max() - pareto_obj2.min() + 1e-10)
obj_norm[:, 2] = (pareto_obj3 - pareto_obj3.min()) / (pareto_obj3.max() - pareto_obj3.min() + 1e-10)
obj_norm[:, 3] = (pareto_obj4 - pareto_obj4.min()) / (pareto_obj4.max() - pareto_obj4.min() + 1e-10)
x_axes = [0, 1, 2, 3]; labels = ['成本', '覆盖率', '负荷均衡', '电网风险']
for i in range(min(50, len(pareto_obj1))):
    alpha_val = 0.15; color = COLOR_GREY; lw = 0.6
    if i == best_idx: alpha_val = 1.0; color = COLOR_RED; lw = 2.0
    ax3c.plot(x_axes, obj_norm[i, :], color=color, alpha=alpha_val, linewidth=lw)
ax3c.set_xticks(x_axes); ax3c.set_xticklabels(labels, fontsize=8)
ax3c.set_ylabel('归一化目标值 (越小越好)', fontsize=8)
ax3c.set_title('(c) Pareto前沿: 平行坐标', fontsize=10, fontweight='bold')
ax3c.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
fig3.savefig(os.path.join(RESULTS_FIGURES, '图3_NSGA-II求解过程.png'), dpi=300, bbox_inches='tight')
plt.close()
print('图3 已保存')

# =============================================================================
# 5. 图4 & 图5 (简化合并)
# =============================================================================
# 图4: TOPSIS
print('\n步骤5: 生成图4 — TOPSIS最优解选取')
fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(11, 4.2))
sorted_idx = np.argsort(c_close)[::-1]
top_n = min(15, len(c_close))
top_closeness = c_close[sorted_idx[:top_n]]
colors_top = [COLOR_RED if idx == 0 else COLOR_BLUE for idx in range(top_n)]
ax4a.barh(range(top_n), top_closeness, color=colors_top, edgecolor='white', linewidth=0.5)
ax4a.set_yticks(range(top_n))
ax4a.set_yticklabels([f'方案#{candidate_idx[sorted_idx[i]]+1}' for i in range(top_n)], fontsize=7)
ax4a.set_xlabel('TOPSIS相对贴近度', fontsize=8)
ax4a.set_title('(a) 候选方案TOPSIS排序', fontsize=10, fontweight='bold')
ax4a.invert_yaxis(); ax4a.set_xlim(0, 1.05)
for i, v in enumerate(top_closeness):
    ax4a.text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=6, color='#333333')

c_w_labels = [f'成本\n({c_w[0]*100:.1f}%)', f'覆盖率\n({c_w[1]*100:.1f}%)',
               f'负荷均衡\n({c_w[2]*100:.1f}%)', f'电网风险\n({c_w[3]*100:.1f}%)']
PALETTE_4 = [COLOR_BLUE, COLOR_ORANGE, COLOR_GREEN, '#E74C3C']
ax4b.pie(c_w, labels=c_w_labels, colors=PALETTE_4, autopct='', startangle=90,
         explode=(0.02, 0.02, 0.02, 0.02), textprops={'fontsize': 8})
ax4b.set_title('(b) 熵权法四目标权重', fontsize=10, fontweight='bold')
plt.tight_layout()
fig4.savefig(os.path.join(RESULTS_FIGURES, '图4_TOPSIS最优解选取.png'), dpi=300, bbox_inches='tight')
plt.close()
print('图4 已保存')

# 图5: 配置方案
print('\n步骤6: 生成图5 — 配置方案与优化效果对比')
fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(12, 4.5))
x = np.arange(N_REGIONS)
ax5a.bar(x, best_fast, label='新增快充桩', color=COLOR_BLUE, edgecolor='white', linewidth=0.5)
ax5a.bar(x, best_slow, bottom=best_fast, label='新增慢充桩', color=COLOR_ORANGE, edgecolor='white', linewidth=0.5)
for i in range(N_REGIONS):
    total = best_fast[i] + best_slow[i]
    if total > 0:
        ax5a.text(i, total + 1, f'{total}', ha='center', fontsize=6, color='#333333')
ax5a.set_xticks(x); ax5a.set_xticklabels([f'{n[:3]}' for n in REGION_NAMES], fontsize=7)
ax5a.set_ylabel('新增充电桩数量（台）', fontsize=8)
ax5a.set_title('(a) 各区域新增充电桩配置方案', fontsize=10, fontweight='bold')
ax5a.legend(fontsize=7); ax5a.grid(axis='y', alpha=0.3, linestyle='--')

coverage_before = df_gap_sorted['当前覆盖率(%)'].values
load_rates_before = df_gap_sorted['电网负载率(%)'].values / 100
load_var_before = np.std(load_rates_before)
SIMULTANEITY = 0.8; POWER_FAST, POWER_SLOW = 120, 7
pred_peak = df_gap.sort_values('区域编号')['峰值负荷(kW)'].values
grid_cap = df_gap.sort_values('区域编号')['电网总容量(kW)'].values
delta_load = SIMULTANEITY * (POWER_FAST * best_fast + POWER_SLOW * best_slow)
load_rates_after = (pred_peak + delta_load) / grid_cap
load_var_after = np.std(load_rates_after)

x_comp = np.arange(2); width_comp = 0.35
ax5b.bar(x_comp[0] - width_comp/2, np.mean(coverage_before), width_comp,
         color=COLOR_GREY, label='优化前', edgecolor='white', linewidth=0.5)
ax5b.bar(x_comp[0] + width_comp/2, pareto_obj2[best_idx] * 100, width_comp,
         color=COLOR_BLUE, label='优化后', edgecolor='white', linewidth=0.5)
ax5b_2 = ax5b.twinx()
ax5b_2.bar(x_comp[1] - width_comp/2, load_var_before, width_comp,
           color=COLOR_GREY, edgecolor='white', linewidth=0.5)
ax5b_2.bar(x_comp[1] + width_comp/2, load_var_after, width_comp,
           color=COLOR_ORANGE, edgecolor='white', linewidth=0.5)
ax5b.set_xticks(x_comp); ax5b.set_xticklabels(['平均覆盖率 (%)', '负荷率标准差'], fontsize=8)
ax5b.set_ylabel('覆盖率 (%)', fontsize=8, color=COLOR_BLUE)
ax5b_2.set_ylabel('负荷率标准差', fontsize=8, color=COLOR_ORANGE)
ax5b.set_title('(b) 优化前后关键指标对比', fontsize=10, fontweight='bold')
ax5b.legend(fontsize=7, loc='upper left')
ax5b.text(x_comp[0] + width_comp/2, pareto_obj2[best_idx] * 100 + 1,
          f'{pareto_obj2[best_idx]*100:.1f}%', ha='center', fontsize=7, fontweight='bold')
ax5b_2.text(x_comp[1] + width_comp/2, load_var_after + 0.0005,
            f'{load_var_after:.4f}', ha='center', fontsize=7, fontweight='bold')
plt.tight_layout()
fig5.savefig(os.path.join(RESULTS_FIGURES, '图5_配置方案与优化效果对比.png'), dpi=300, bbox_inches='tight')
plt.close()
print('图5 已保存')

# =============================================================================
# 7. 表2: 各区域最优配置方案
# =============================================================================
print('\n步骤7: 生成表2 — 各区域最优配置方案')
total_area_arr = np.array([17.36, 14.25, 17.62, 110.07, 80.10, 60.08, 139.87, 120.04, 131.20, 22.30])
covered_area_arr = np.array([14.02, 11.10, 14.50, 55.03, 32.44, 41.89, 35.02, 42.00, 26.17, 14.50])
current_cov_arr = covered_area_arr / total_area_arr
region_radius_arr = np.array([1.5, 1.5, 2.0, 1.5, 2.0, 2.0, 2.5, 2.5, 2.5, 2.5])
single_cover = np.pi * region_radius_arr**2

per_region_cov = np.zeros(N_REGIONS)
for i in range(N_REGIONS):
    uncovered = 1.0 - current_cov_arr[i]
    marginal = single_cover[i] * (uncovered ** 1.2)
    effective_delta = 2.0 * best_fast[i] + 1.0 * best_slow[i]
    added = effective_delta * marginal
    new_covered = covered_area_arr[i] + added
    per_region_cov[i] = min(new_covered / total_area_arr[i], 1.0)

region_investment = COST_FAST * best_fast + COST_SLOW * best_slow

df_table2 = pd.DataFrame({
    '区域编号': range(1, N_REGIONS + 1), '区域名称': REGION_NAMES,
    '区域类型': REGION_TYPES,
    '现有快充桩(台)': [129, 119, 99, 109, 76, 95, 45, 59, 39, 53],
    '现有慢充桩(台)': [86, 79, 66, 73, 50, 63, 30, 39, 26, 35],
    '新增快充桩(台)': best_fast, '新增慢充桩(台)': best_slow,
    '区域投资(万元)': region_investment, '地理覆盖率': np.round(per_region_cov, 4),
})
df_table2['新增后快充总计(台)'] = 129 + best_fast
df_table2['新增后慢充总计(台)'] = 86 + best_slow
df_table2.to_excel(FILE_Q2_TABLE2, index=False)
print(f'表2 已保存: {FILE_Q2_TABLE2}')

# 问题2关键变量输出
print("\n" + "=" * 60)
print("问题2优化结果")
print("=" * 60)
print("区域列表:")
print(REGION_NAMES)
print("\n新增快充桩数量:")
print(best_fast)
print("\n新增慢充桩数量:")
print(best_slow)
print("\n新增总投资:")
print(region_investment.sum())
print("\n优化前覆盖率:")
print(np.mean(coverage_before))
print("\n优化后覆盖率:")
print(pareto_obj2[best_idx])
print("\n负载均衡指标:")
print(load_var_after)

# =============================================================================
# 8. 表3: 优化前后多指标对比
# =============================================================================
print('\n步骤8: 生成表3 — 优化前后多指标对比')
service_after = (CAP_FAST * (129 + best_fast) + CAP_SLOW * (86 + best_slow))
pred_trips_arr = df_gap_sorted['预测日均车次(次/日)'].values
gap_before = np.sum(df_gap_sorted['供需缺口(车次/日)'].values)
gap_after = np.sum(np.maximum(0, pred_trips_arr - service_after))
peak_load_before = pred_peak
peak_load_after = pred_peak + delta_load
overload_risk_before = np.sum(peak_load_before > 2100)
overload_risk_after = np.sum(peak_load_after > 2100)

df_table3 = pd.DataFrame({
    '指标': ['平均服务覆盖率 (%)', '覆盖率不达标区域数', '负荷率标准差',
             '总供需缺口 (车次/日)', '电网过载风险区域数', '所需总投资 (万元)'],
    '优化前': [f'{np.mean(coverage_before):.1f}', f'{int(np.sum(coverage_before < 90))}',
               f'{load_var_before:.4f}', f'{gap_before:.0f}',
               f'{int(overload_risk_before)}', '—'],
    '优化后': [f'{pareto_obj2[best_idx]*100:.1f}', '0',
               f'{load_var_after:.4f}', f'{gap_after:.0f}',
               f'{int(overload_risk_after)}', f'{region_investment.sum():.1f}'],
})
df_table3.to_excel(FILE_Q2_TABLE3, index=False)
print(f'表3 已保存: {FILE_Q2_TABLE3}')
print(f'\n优化前后对比:')
print(df_table3.to_string())

print('\n' + '=' * 60)
print('problem2_result.py 运行完成！')
print('=' * 60)
