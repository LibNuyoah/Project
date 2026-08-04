"""
=============================================================================
problem2_result.py — 问题二：结果分析与可视化
=============================================================================
功能：
  1. 熵权-TOPSIS方法从Pareto前沿中选取最优折中方案
  2. 生成全部5张合并图表和3张数据表
  3. 输出最终配置方案和优化前后对比

输入文件：
  - output/preprocess_data.npz
  - output/optimization_result.npz
  - output/Pareto前沿解集.xlsx
  - output/表1_各区域供需缺口与建设紧迫度.xlsx

输出文件（图表）：
  - output/图1_建设紧迫度与供需缺口.png
  - output/图2_空间溢出权重热力图.png
  - output/图3_NSGA-II求解过程.png
  - output/图4_TOPSIS最优解选取.png
  - output/图5_配置方案与优化效果对比.png

输出文件（表格）：
  - output/表2_各区域最优配置方案.xlsx
  - output/表3_优化前后多指标对比.xlsx

=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import os, sys
import warnings
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
OUTPUT_DIR = os.path.join(ROOT, 'result', 'tables')
FIGURES_DIR = os.path.join(ROOT, 'result', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# 全局绘图风格设置
# =============================================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.size': 9,
})

# 统一配色（蓝色系为主，辅助色用于对比）
COLOR_BLUE = '#2B579A'
COLOR_ORANGE = '#E07B39'
COLOR_GREEN = '#3A8E6F'
COLOR_RED = '#C44E52'
COLOR_GREY = '#8C8C8C'
PALETTE_3 = [COLOR_BLUE, COLOR_ORANGE, COLOR_GREEN]
PALETTE_10 = sns.color_palette('Blues', 10)

# 10个区域名称（按编号顺序）
REGION_NAMES = ['宝塔山街道', '南市街道', '凤凰山街道', '枣园街道', '桥沟街道',
                '新城街道', '柳林镇', '河庄坪镇', '姚店镇', '李渠镇']
REGION_TYPES = ['老城核心区', '老城核心区', '城市新区', '老城核心区', '城市新区',
                '城市新区', '城郊/工业区', '城郊/工业区', '城郊/工业区', '城郊/工业区']

N_REGIONS = 10
os.makedirs('output', exist_ok=True)

# =============================================================================
# 0. 加载数据
# =============================================================================
print('=' * 60)
print('问题二 结果分析与可视化')
print('=' * 60)

# 预处理数据
data_pre = np.load(os.path.join(OUTPUT_DIR, 'preprocess_data.npz'), allow_pickle=True)
spillover_matrix = data_pre['spillover_matrix']
urgency_weights = data_pre['urgency_weights']

# 优化结果
data_opt = np.load(os.path.join(OUTPUT_DIR, 'optimization_result.npz'), allow_pickle=True)
pareto_obj1 = data_opt['pareto_obj1']   # 成本
pareto_obj2 = data_opt['pareto_obj2']   # 覆盖率
pareto_obj3 = data_opt['pareto_obj3']   # 负荷率方差
pareto_fast = data_opt['pareto_fast']   # 新增快充
pareto_slow = data_opt['pareto_slow']   # 新增慢充
convergence = data_opt['convergence_history'].item()

df_gap = pd.read_excel(os.path.join(OUTPUT_DIR, '表1_各区域供需缺口与建设紧迫度.xlsx'))

# 附件5参数
COST_FAST, COST_SLOW = 6.0, 0.8
CAP_FAST, CAP_SLOW = 80, 20

print(f'Pareto解集大小: {len(pareto_obj1)}')
print(f'收敛曲线记录: {len(convergence["generation"])} 代')

# =============================================================================
# 1. 熵权-TOPSIS 最优解选取
# =============================================================================
print('\n' + '=' * 60)
print('步骤1: 熵权-TOPSIS选取最优折中方案')
print('=' * 60)


def entropy_weight_topsis(obj_matrix):
    """
    熵权法 + TOPSIS 从Pareto前沿中选取最优方案

    参数:
        obj_matrix: shape (n_solutions, 3)
                    列0=成本(↓), 列1=覆盖率(↑), 列2=方差(↓)

    返回:
        best_idx: 最优方案索引
        weights: 三目标熵权权重
        closeness: 每个方案的TOPSIS贴近度
    """
    n, m = obj_matrix.shape

    # ---- 步骤1: 归一化（区分正向/负向指标） ----
    norm_matrix = np.zeros((n, m))
    directions = ['cost', 'benefit', 'cost']  # 成本↓, 覆盖率↑, 方差↓

    for j in range(m):
        col = obj_matrix[:, j]
        col_min, col_max = col.min(), col.max()
        if col_max > col_min:
            if directions[j] == 'benefit':
                norm_matrix[:, j] = (col - col_min) / (col_max - col_min)
            else:
                norm_matrix[:, j] = (col_max - col) / (col_max - col_min)
        else:
            norm_matrix[:, j] = 0.5

    # ---- 步骤2: 熵权法 ----
    p = np.clip(norm_matrix / norm_matrix.sum(axis=0, keepdims=True), 1e-10, 1)
    e = -np.sum(p * np.log(p), axis=0) / np.log(n)
    w = (1 - e) / np.sum(1 - e)

    # ---- 步骤3: TOPSIS ----
    weighted = norm_matrix * w
    ideal_pos = weighted.max(axis=0)    # 正理想解
    ideal_neg = weighted.min(axis=0)    # 负理想解

    d_pos = np.sqrt(np.sum((weighted - ideal_pos)**2, axis=1))
    d_neg = np.sqrt(np.sum((weighted - ideal_neg)**2, axis=1))

    closeness = d_neg / (d_pos + d_neg)

    best_idx = np.argmax(closeness)
    return best_idx, w, closeness


# 构建评价矩阵
obj_matrix = np.column_stack([pareto_obj1, pareto_obj2, pareto_obj3])
best_idx, topsis_weights, closeness = entropy_weight_topsis(obj_matrix)

print(f'\n熵权法确定的权重:')
print(f'  成本(Z1):     {topsis_weights[0]:.4f}')
print(f'  覆盖率(Z2):   {topsis_weights[1]:.4f}')
print(f'  负荷均衡(Z3): {topsis_weights[2]:.4f}')

print(f'\n最优方案索引: #{best_idx+1} (共{len(closeness)}个Pareto解)')
print(f'TOPSIS贴近度: {closeness[best_idx]:.4f}')
print(f'  成本: {pareto_obj1[best_idx]:.1f} 万元')
print(f'  覆盖率: {pareto_obj2[best_idx]:.4f} ({pareto_obj2[best_idx]*100:.1f}%)')
print(f'  负荷率方差: {pareto_obj3[best_idx]:.6f}')

# 最优方案的决策变量
best_fast = pareto_fast[best_idx].astype(int)
best_slow = pareto_slow[best_idx].astype(int)

# =============================================================================
# 2. 图1: 建设紧迫度与供需缺口（双栏图）
# =============================================================================
print('\n' + '=' * 60)
print('步骤2: 生成图1 — 建设紧迫度与供需缺口')
print('=' * 60)

fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(12, 4.5))

# 按紧迫度排序
df_gap_sorted = df_gap.sort_values('建设紧迫度指数(0-100)', ascending=True)

# (a) 建设紧迫度水平柱状图
colors_urgency = [COLOR_RED if v > 60 else (COLOR_ORANGE if v > 30 else COLOR_BLUE)
                  for v in df_gap_sorted['建设紧迫度指数(0-100)']]
bars1 = ax1a.barh(range(N_REGIONS), df_gap_sorted['建设紧迫度指数(0-100)'],
                   color=colors_urgency, edgecolor='white', linewidth=0.5)
ax1a.set_yticks(range(N_REGIONS))
ax1a.set_yticklabels(df_gap_sorted['区域名称'], fontsize=7)
ax1a.set_xlabel('建设紧迫度指数 (0-100)', fontsize=8)
ax1a.set_title('(a) 各区域建设紧迫度排序', fontsize=10, fontweight='bold')
ax1a.invert_yaxis()
# 添加数值标签
for i, (v, name) in enumerate(zip(df_gap_sorted['建设紧迫度指数(0-100)'],
                                    df_gap_sorted['区域名称'])):
    ax1a.text(v + 1, i, f'{v:.0f}', va='center', fontsize=6, color='#333333')

# (b) 供需缺口 vs 现有能力
x = np.arange(N_REGIONS)
width = 0.35
existing = df_gap_sorted['现有服务能力(车次/日)'].values
demand = df_gap_sorted['预测日均车次(次/日)'].values

bars_exist = ax1b.bar(x - width/2, existing, width, label='现有服务能力',
                       color=COLOR_BLUE, edgecolor='white', linewidth=0.5)
bars_demand = ax1b.bar(x + width/2, demand, width, label='预测车次需求',
                        color=COLOR_ORANGE, edgecolor='white', linewidth=0.5)
ax1b.set_xticks(x)
ax1b.set_xticklabels(df_gap_sorted['区域名称'], rotation=30, ha='right', fontsize=7)
ax1b.set_ylabel('车次/日', fontsize=8)
ax1b.set_title('(b) 各区域充电供需对比', fontsize=10, fontweight='bold')
ax1b.legend(fontsize=7, loc='upper left')
ax1b.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
fig1.savefig(os.path.join(OUTPUT_DIR, '图1_建设紧迫度与供需缺口.png'), dpi=300, bbox_inches='tight')
plt.close()
print('图1 已保存: output/图1_建设紧迫度与供需缺口.png')

# =============================================================================
# 3. 图2: 空间溢出权重热力图
# =============================================================================
print('\n步骤3: 生成图2 — 空间溢出权重热力图')

fig2, ax2 = plt.subplots(figsize=(7, 6))

# 标注溢出矩阵
annot_matrix = np.zeros_like(spillover_matrix)
for i in range(N_REGIONS):
    for j in range(N_REGIONS):
        annot_matrix[i, j] = spillover_matrix[i, j]

sns.heatmap(spillover_matrix, annot=np.round(annot_matrix, 2),
            fmt='.2f', cmap='YlOrRd', ax=ax2,
            xticklabels=[f'{n[:2]}' for n in REGION_NAMES],
            yticklabels=[f'R{i+1}-{n[:2]}' for i, n in enumerate(REGION_NAMES)],
            vmin=0, vmax=1, linewidths=0.5, linecolor='white',
            cbar_kws={'label': '溢出权重', 'shrink': 0.8})

ax2.set_title('空间溢出权重矩阵 W (行i→列j: 区域i的桩服务区域j)', fontsize=10, fontweight='bold')
ax2.set_xlabel('被服务区域 j', fontsize=9)
ax2.set_ylabel('充电桩所在区域 i', fontsize=9)

plt.tight_layout()
fig2.savefig(os.path.join(OUTPUT_DIR, '图2_空间溢出权重热力图.png'), dpi=300, bbox_inches='tight')
plt.close()
print('图2 已保存: output/图2_空间溢出权重热力图.png')

# =============================================================================
# 4. 图3: NSGA-II求解过程（三栏图）
# =============================================================================
print('\n步骤4: 生成图3 — NSGA-II求解过程')

fig3 = plt.figure(figsize=(14, 4.2))

# (a) 收敛曲线
ax3a = fig3.add_subplot(1, 3, 1)
gens = convergence['generation']
ax3a.plot(gens, convergence['min_cost'], color=COLOR_BLUE, linewidth=1.2, label='最小成本')
ax3a.set_xlabel('迭代代数', fontsize=8)
ax3a.set_ylabel('最小成本 (万元)', fontsize=8, color=COLOR_BLUE)
ax3a.tick_params(axis='y', labelcolor=COLOR_BLUE)
ax3a_2 = ax3a.twinx()
ax3a_2.plot(gens, convergence['max_coverage'], color=COLOR_ORANGE, linewidth=1.2, label='最大覆盖率')
ax3a_2.set_ylabel('最大覆盖率', fontsize=8, color=COLOR_ORANGE)
ax3a_2.tick_params(axis='y', labelcolor=COLOR_ORANGE)
ax3a.set_title('(a) NSGA-II 收敛曲线', fontsize=10, fontweight='bold')
ax3a.grid(alpha=0.3, linestyle='--')

# (b) Pareto前沿 — 成本 vs 覆盖率
ax3b = fig3.add_subplot(1, 3, 2)
scatter_b = ax3b.scatter(pareto_obj1, pareto_obj2 * 100, c=pareto_obj3,
                          cmap='YlOrRd', s=25, edgecolors='grey', linewidth=0.3, alpha=0.8)
ax3b.set_xlabel('总成本 (万元)', fontsize=8)
ax3b.set_ylabel('平均覆盖率 (%)', fontsize=8)
ax3b.set_title('(b) Pareto前沿: 成本 vs 覆盖率', fontsize=10, fontweight='bold')
ax3b.grid(alpha=0.3, linestyle='--')
cbar_b = plt.colorbar(scatter_b, ax=ax3b, shrink=0.8)
cbar_b.set_label('负荷率方差', fontsize=7)

# 标注最优方案
ax3b.scatter(pareto_obj1[best_idx], pareto_obj2[best_idx] * 100,
             color=COLOR_RED, s=80, marker='*', edgecolors='white',
             linewidth=1.5, zorder=5, label='TOPSIS最优解')
ax3b.legend(fontsize=7, loc='lower right')

# (c) 三目标平行坐标图
ax3c = fig3.add_subplot(1, 3, 3)
# 标准化三目标到[0,1]用于平行坐标
obj_norm = np.zeros_like(obj_matrix)
obj_norm[:, 0] = (obj_matrix[:, 0] - obj_matrix[:, 0].min()) / (obj_matrix[:, 0].max() - obj_matrix[:, 0].min() + 1e-10)
obj_norm[:, 1] = (obj_matrix[:, 1].max() - obj_matrix[:, 1]) / (obj_matrix[:, 1].max() - obj_matrix[:, 1].min() + 1e-10)  # 覆盖率反转（高覆盖率=好）
obj_norm[:, 2] = (obj_matrix[:, 2] - obj_matrix[:, 2].min()) / (obj_matrix[:, 2].max() - obj_matrix[:, 2].min() + 1e-10)

x_axes = [0, 1, 2]
labels = ['成本', '覆盖率', '负荷均衡']
for i in range(min(50, len(pareto_obj1))):  # 最多画50条避免过密
    alpha_val = 0.15
    color = COLOR_GREY
    lw = 0.6
    if i == best_idx:
        alpha_val = 1.0
        color = COLOR_RED
        lw = 2.0
    ax3c.plot(x_axes, obj_norm[i, :], color=color, alpha=alpha_val, linewidth=lw)

ax3c.set_xticks(x_axes)
ax3c.set_xticklabels(labels, fontsize=8)
ax3c.set_ylabel('归一化目标值 (越小越好)', fontsize=8)
ax3c.set_title('(c) Pareto前沿: 平行坐标', fontsize=10, fontweight='bold')
ax3c.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
fig3.savefig(os.path.join(OUTPUT_DIR, '图3_NSGA-II求解过程.png'), dpi=300, bbox_inches='tight')
plt.close()
print('图3 已保存: output/图3_NSGA-II求解过程.png')

# =============================================================================
# 5. 图4: TOPSIS最优解选取（双栏图）
# =============================================================================
print('\n步骤5: 生成图4 — TOPSIS最优解选取')

fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(11, 4.2))

# (a) TOPSIS贴近度排序
sorted_idx = np.argsort(closeness)[::-1]
top_n = min(15, len(closeness))
top_closeness = closeness[sorted_idx[:top_n]]
colors_top = [COLOR_RED if idx == 0 else COLOR_BLUE for idx in range(top_n)]

bars4 = ax4a.barh(range(top_n), top_closeness, color=colors_top, edgecolor='white', linewidth=0.5)
ax4a.set_yticks(range(top_n))
ax4a.set_yticklabels([f'方案#{sorted_idx[i]+1}' for i in range(top_n)], fontsize=7)
ax4a.set_xlabel('TOPSIS相对贴近度', fontsize=8)
ax4a.set_title('(a) Pareto方案TOPSIS排序 (TOP15)', fontsize=10, fontweight='bold')
ax4a.invert_yaxis()
for i, v in enumerate(top_closeness):
    ax4a.text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=6, color='#333333')
ax4a.set_xlim(0, 1.05)

# (b) 熵权法权重饼图
weights_labels = [f'成本\n({topsis_weights[0]*100:.1f}%)',
                  f'覆盖率\n({topsis_weights[1]*100:.1f}%)',
                  f'负荷均衡\n({topsis_weights[2]*100:.1f}%)']
wedges, texts, autotexts = ax4b.pie(
    topsis_weights, labels=weights_labels, colors=PALETTE_3,
    autopct='', startangle=90, explode=(0.02, 0.02, 0.02),
    textprops={'fontsize': 8})
ax4b.set_title('(b) 熵权法确定的目标权重', fontsize=10, fontweight='bold')

plt.tight_layout()
fig4.savefig(os.path.join(OUTPUT_DIR, '图4_TOPSIS最优解选取.png'), dpi=300, bbox_inches='tight')
plt.close()
print('图4 已保存: output/图4_TOPSIS最优解选取.png')

# =============================================================================
# 6. 图5: 配置方案与优化效果对比（双栏图）
# =============================================================================
print('\n步骤6: 生成图5 — 配置方案与优化效果对比')

fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(12, 4.5))

# (a) 各区域新增快充/慢充堆叠柱状图
x = np.arange(N_REGIONS)
ax5a.bar(x, best_fast, label='新增快充桩', color=COLOR_BLUE, edgecolor='white', linewidth=0.5)
ax5a.bar(x, best_slow, bottom=best_fast, label='新增慢充桩', color=COLOR_ORANGE,
         edgecolor='white', linewidth=0.5)
# 在柱子上标注总数
for i in range(N_REGIONS):
    total = best_fast[i] + best_slow[i]
    if total > 0:
        ax5a.text(i, total + 1, f'{total}', ha='center', fontsize=6, color='#333333')
ax5a.set_xticks(x)
ax5a.set_xticklabels([f'{n[:3]}' for n in REGION_NAMES], fontsize=7)
ax5a.set_ylabel('新增充电桩数量（台）', fontsize=8)
ax5a.set_title('(a) 各区域新增充电桩配置方案', fontsize=10, fontweight='bold')
ax5a.legend(fontsize=7)
ax5a.grid(axis='y', alpha=0.3, linestyle='--')

# (b) 优化前后对比 — 覆盖率 + 负荷率标准差
# 优化前数据
coverage_before = df_gap_sorted['当前覆盖率(%)'].values
# 优化前负荷率标准差 (用df_gap数据估算)
df_gap_sorted_original = df_gap.sort_values('建设紧迫度指数(0-100)', ascending=True)
load_rates_before = df_gap_sorted_original['电网负载率(%)'].values / 100
load_var_before = np.std(load_rates_before)

# 优化后各区域覆盖率
coverage_after = np.full(N_REGIONS, pareto_obj2[best_idx] * 100)  # 平均覆盖率
# 优化后负荷率（从最优方案的delta_fast/delta_slow推算）
SIMULTANEITY = 0.8
POWER_FAST, POWER_SLOW = 120, 7
pred_peak = df_gap_sorted_original['峰值负荷(kW)'].values
grid_cap = df_gap_sorted_original['电网总容量(kW)'].values
# 需要按紧迫度排序对应的最优解
delta_load = SIMULTANEITY * (POWER_FAST * best_fast + POWER_SLOW * best_slow)
load_rates_after = (pred_peak + delta_load) / grid_cap
load_var_after = np.std(load_rates_after)

x_comp = np.arange(2)
width_comp = 0.35

ax5b.bar(x_comp[0] - width_comp/2, np.mean(coverage_before), width_comp,
         color=COLOR_GREY, label='优化前', edgecolor='white', linewidth=0.5)
ax5b.bar(x_comp[0] + width_comp/2, pareto_obj2[best_idx] * 100, width_comp,
         color=COLOR_BLUE, label='优化后', edgecolor='white', linewidth=0.5)

ax5b_2 = ax5b.twinx()
ax5b_2.bar(x_comp[1] - width_comp/2, load_var_before, width_comp,
           color=COLOR_GREY, edgecolor='white', linewidth=0.5)
ax5b_2.bar(x_comp[1] + width_comp/2, load_var_after, width_comp,
           color=COLOR_ORANGE, edgecolor='white', linewidth=0.5)

ax5b.set_xticks(x_comp)
ax5b.set_xticklabels(['平均覆盖率 (%)', '负荷率标准差'], fontsize=8)
ax5b.set_ylabel('覆盖率 (%)', fontsize=8, color=COLOR_BLUE)
ax5b_2.set_ylabel('负荷率标准差', fontsize=8, color=COLOR_ORANGE)
ax5b.set_title('(b) 优化前后关键指标对比', fontsize=10, fontweight='bold')
ax5b.legend(fontsize=7, loc='upper left')

# 标注数值
ax5b.text(x_comp[0] + width_comp/2, pareto_obj2[best_idx] * 100 + 1,
          f'{pareto_obj2[best_idx]*100:.1f}%', ha='center', fontsize=7, fontweight='bold')
ax5b_2.text(x_comp[1] + width_comp/2, load_var_after + 0.0005,
            f'{load_var_after:.4f}', ha='center', fontsize=7, fontweight='bold')

plt.tight_layout()
fig5.savefig(os.path.join(OUTPUT_DIR, '图5_配置方案与优化效果对比.png'), dpi=300, bbox_inches='tight')
plt.close()
print('图5 已保存: output/图5_配置方案与优化效果对比.png')

# =============================================================================
# 7. 表2: 各区域最优配置方案
# =============================================================================
print('\n步骤7: 生成表2 — 各区域最优配置方案')

df_table2 = pd.DataFrame({
    '区域编号': range(1, N_REGIONS + 1),
    '区域名称': REGION_NAMES,
    '区域类型': REGION_TYPES,
    '现有快充桩(台)': [129, 119, 99, 109, 76, 95, 45, 59, 39, 53],
    '现有慢充桩(台)': [86, 79, 66, 73, 50, 63, 30, 39, 26, 35],
    '新增快充桩(台)': best_fast,
    '新增慢充桩(台)': best_slow,
    '新增后快充总计(台)': [129 + f for f in best_fast],
    '新增后慢充总计(台)': [86 + s for s in best_slow],
})

# 计算各区域投资
region_investment = COST_FAST * best_fast + COST_SLOW * best_slow
df_table2['区域投资(万元)'] = region_investment

# 各区域优化后服务能力
service_after = (CAP_FAST * (df_table2['现有快充桩(台)'] + best_fast) +
                 CAP_SLOW * (df_table2['现有慢充桩(台)'] + best_slow))
df_table2['优化后服务能力(车次/日)'] = service_after

# 覆盖率贡献
pred_trips = df_gap_sorted['预测日均车次(次/日)'].values
df_table2['区域覆盖率'] = np.minimum(service_after.values / pred_trips, 1.0)

output_table2 = os.path.join(OUTPUT_DIR, '表2_各区域最优配置方案.xlsx')
df_table2.to_excel(output_table2, index=False)
print(f'表2 已保存: {output_table2}')
print(f'\n总投资: {region_investment.sum():.1f} 万元')
print(f'快充桩合计新增: {best_fast.sum()} 台')
print(f'慢充桩合计新增: {best_slow.sum()} 台')

# =============================================================================
# 8. 表3: 优化前后多指标对比
# =============================================================================
print('\n步骤8: 生成表3 — 优化前后多指标对比')

# 优化前指标
coverage_before_arr = df_gap_sorted['当前覆盖率(%)'].values
mean_cov_before = np.mean(coverage_before_arr) / 100
below_90_before = np.sum(coverage_before_arr < 90)
load_rates_before_arr = df_gap_sorted['电网负载率(%)'].values / 100
load_std_before = np.std(load_rates_before_arr)
peak_load_before = df_gap_sorted['峰值负荷(kW)'].values
overload_risk_before = np.sum(peak_load_before > 2100)

# 优化后指标
mean_cov_after = pareto_obj2[best_idx]
below_90_after = 0  # 约束保证全覆盖≥90%
load_std_after = load_var_after
peak_load_after = pred_peak + delta_load
overload_risk_after = np.sum(peak_load_after > 2100)

# 计算覆盖缺口（预测需求 vs 服务能力）
gap_before = np.sum(df_gap_sorted['供需缺口(车次/日)'].values)
gap_after = np.sum(np.maximum(0, pred_trips - service_after.values))

df_table3 = pd.DataFrame({
    '指标': [
        '平均服务覆盖率 (%)',
        '覆盖率不达标区域数',
        '负荷率标准差',
        '总供需缺口 (车次/日)',
        '电网过载风险区域数',
        '所需总投资 (万元)',
    ],
    '优化前': [
        f'{mean_cov_before*100:.1f}',
        f'{below_90_before}',
        f'{load_std_before:.4f}',
        f'{gap_before:.0f}',
        f'{overload_risk_before}',
        '—',
    ],
    '优化后': [
        f'{mean_cov_after*100:.1f}',
        f'{below_90_after}',
        f'{load_std_after:.4f}',
        f'{gap_after:.0f}',
        f'{overload_risk_after}',
        f'{region_investment.sum():.1f}',
    ],
})

output_table3 = os.path.join(OUTPUT_DIR, '表3_优化前后多指标对比.xlsx')
df_table3.to_excel(output_table3, index=False)
print(f'表3 已保存: {output_table3}')
print(f'\n优化前后对比:')
print(df_table3.to_string())

# =============================================================================
# 9. 汇总输出清单
# =============================================================================
print('\n' + '=' * 60)
print('全部输出文件清单')
print('=' * 60)
output_files = [
    os.path.join(OUTPUT_DIR, '图1_建设紧迫度与供需缺口.png'),
    os.path.join(OUTPUT_DIR, '图2_空间溢出权重热力图.png'),
    os.path.join(OUTPUT_DIR, '图3_NSGA-II求解过程.png'),
    os.path.join(OUTPUT_DIR, '图4_TOPSIS最优解选取.png'),
    os.path.join(OUTPUT_DIR, '图5_配置方案与优化效果对比.png'),
    os.path.join(OUTPUT_DIR, '表1_各区域供需缺口与建设紧迫度.xlsx'),
    os.path.join(OUTPUT_DIR, '表2_各区域最优配置方案.xlsx'),
    os.path.join(OUTPUT_DIR, '表3_优化前后多指标对比.xlsx'),
    os.path.join(OUTPUT_DIR, 'Pareto前沿解集.xlsx'),
    os.path.join(OUTPUT_DIR, 'NSGA-II收敛曲线数据.xlsx'),
]
for f in output_files:
    exists = '✓' if os.path.exists(f) else '✗'
    print(f'  [{exists}] {f}')

print('\n' + '=' * 60)
print('problem2_result.py 运行完成！')
print('=' * 60)
