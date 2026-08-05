"""
=============================================================================
problem2_data.py — 问题二：数据预处理与供需分析
=============================================================================
功能：
  1. 加载附件1-5及问题1预测结果
  2. 计算各区域充电供需缺口
  3. 计算电网剩余容量
  4. 构建建设紧迫度指数
  5. 估算区域间距离矩阵与空间溢出权重矩阵
  6. 输出 表1_各区域供需缺口与建设紧迫度.xlsx

输入文件：
  - data/raw/附件1-5
  - results/prediction_result.xlsx (问题1输出)
  - results/tables/hourly_prediction.xlsx (问题1输出)

输出文件：
  - results/tables/表1_各区域供需缺口与建设紧迫度.xlsx
  - results/tables/空间溢出权重矩阵.xlsx
=============================================================================
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from utils.paths import (
    FILE_ATTACHMENT1, FILE_ATTACHMENT4,
    FILE_PREDICTION_RESULT, FILE_HOURLY_PREDICTION,
    RESULTS_TABLES, FILE_Q2_TABLE1, FILE_Q2_SPILLOVER, FILE_Q2_DISTANCE,
    FILE_Q2_PREPROCESS
)

# =============================================================================
# 1. 加载原始数据
# =============================================================================
print('=' * 60)
print('步骤1: 加载原始数据')
print('=' * 60)

df_annex1 = pd.read_excel(FILE_ATTACHMENT1, nrows=10)
print(f'附件1: {df_annex1.shape[0]} 个区域, {df_annex1.shape[1]} 个字段')

df_annex4 = pd.read_excel(FILE_ATTACHMENT4)
print(f'附件4: {df_annex4.shape[0]} 个区域, {df_annex4.shape[1]} 列')

df_pred_daily = pd.read_excel(FILE_PREDICTION_RESULT)
print(f'prediction_result: {df_pred_daily.shape[0]} 个区域')

df_pred_hourly = pd.read_excel(FILE_HOURLY_PREDICTION)
print(f'hourly_prediction: {df_pred_hourly.shape[0]} 条记录')

# =============================================================================
# 2. 附件5参数
# =============================================================================
print('\n' + '=' * 60)
print('步骤2: 加载附件5参数')
print('=' * 60)

COST_FAST = 6.0; COST_SLOW = 0.8
CAP_FAST = 80; CAP_SLOW = 20
POWER_FAST = 120; POWER_SLOW = 7
R_CORE = 1.5; R_NEW = 2.0; R_SUBURB = 2.5
COVERAGE_MIN = 0.90
OVERLOAD_THRESHOLD = 2100
SIMULTANEITY = 0.8
AVG_CHARGE_PER_CAR = 12.0

print(f'快充桩: {COST_FAST}万元/台, {CAP_FAST}车次/日, {POWER_FAST}kW')
print(f'慢充桩: {COST_SLOW}万元/台, {CAP_SLOW}车次/日, {POWER_SLOW}kW')
print(f'服务半径: 核心{R_CORE}km / 新区{R_NEW}km / 城郊{R_SUBURB}km')
print(f'最低覆盖率: {COVERAGE_MIN*100}%')
print(f'单车平均充电: {AVG_CHARGE_PER_CAR} kWh')

# =============================================================================
# 3. 各区域供需缺口计算
# =============================================================================
print('\n' + '=' * 60)
print('步骤3: 计算各区域充电供需缺口')
print('=' * 60)

df_gap = df_pred_daily[['区域编号', '区域名称', '区域类型',
                         '预测日均需求_kWh', '预测日均需求_MWh',
                         '工作日日均需求_kWh', '周末日均需求_kWh',
                         '峰值负荷_kWh']].copy()

df_annex1_select = df_annex1[['区域编号', '区域总面积(km²)', '充电覆盖面积面积(km²)',
                               '现有充电桩数量（个）', '其中快充桩（个）', '其中慢充桩（个）',
                               '区域电网总容量（万千瓦）']].copy()
df_annex1_select.columns = ['区域编号', '区域总面积_km2', '充电覆盖面积_km2',
                             '现有桩总数', '现有快充桩', '现有慢充桩',
                             '电网总容量_万kW']

df_gap = df_gap.merge(df_annex1_select, on='区域编号', how='left')
df_gap['现有服务能力_车次日'] = (df_gap['现有快充桩'] * CAP_FAST +
                               df_gap['现有慢充桩'] * CAP_SLOW)
df_gap['预测日均车次'] = df_gap['预测日均需求_kWh'] / AVG_CHARGE_PER_CAR
df_gap['供需缺口_车次日'] = np.maximum(0, df_gap['预测日均车次'] - df_gap['现有服务能力_车次日'])
df_gap['缺口率'] = df_gap['供需缺口_车次日'] / df_gap['现有服务能力_车次日']
df_gap['当前覆盖率'] = df_gap['充电覆盖面积_km2'] / df_gap['区域总面积_km2']
df_gap['电网总容量_kW'] = df_gap['电网总容量_万kW'] * 10000
df_gap['电网负载率'] = df_gap['峰值负荷_kWh'] / df_gap['电网总容量_kW']

print('各区域供需缺口概况：')
print(df_gap[['区域名称', '现有服务能力_车次日', '预测日均车次',
              '供需缺口_车次日', '缺口率', '当前覆盖率', '电网负载率']].to_string())

# =============================================================================
# 4. 电网剩余容量计算
# =============================================================================
print('\n' + '=' * 60)
print('步骤4: 计算电网剩余容量')
print('=' * 60)

df_hourly = df_pred_hourly.copy()
hour_cols_annex4 = [c for c in df_annex4.columns if '-' in str(c)]

grid_capacity_records = []
for _, row in df_annex4.iterrows():
    region_id = int(row['区域']) if '区域' in df_annex4.columns else int(row.iloc[1])
    for h_idx, h_col in enumerate(hour_cols_annex4):
        grid_capacity_records.append({
            '区域编号': region_id, '小时': h_idx, '电网允许负荷_kW': row[h_col]
        })
df_grid = pd.DataFrame(grid_capacity_records)
df_grid['区域编号'] = df_grid['区域编号'].astype(int)

df_compare = df_hourly.merge(df_grid, on=['区域编号', '小时'], how='left')
df_compare['剩余容量_kW'] = df_compare['电网允许负荷_kW'] - df_compare['预测负荷']

df_min_remain = df_compare.groupby('区域编号')['剩余容量_kW'].min().reset_index()
df_min_remain.columns = ['区域编号', '最小剩余容量_kW']

df_peak_remain = df_compare.loc[df_compare.groupby('区域编号')['预测负荷'].idxmax(),
                                 ['区域编号', '小时', '预测负荷', '电网允许负荷_kW', '剩余容量_kW']]
df_peak_remain.columns = ['区域编号', '峰值时段', '峰值预测负荷_kW',
                           '峰值时段电网允许_kW', '峰值时段剩余容量_kW']

df_gap = df_gap.merge(df_min_remain, on='区域编号', how='left')
df_gap = df_gap.merge(df_peak_remain[['区域编号', '峰值时段', '峰值时段剩余容量_kW']],
                       on='区域编号', how='left')

df_gap['过载风险'] = df_gap['最小剩余容量_kW'].apply(
    lambda x: '高风险' if x < -200 else ('中风险' if x < 0 else ('低风险' if x < 500 else '安全'))
)

print('各区域电网剩余容量与过载风险：')
print(df_gap[['区域名称', '峰值负荷_kWh', '电网总容量_kW',
              '最小剩余容量_kW', '峰值时段', '过载风险']].to_string())

# =============================================================================
# 5. 建设紧迫度指数
# =============================================================================
print('\n' + '=' * 60)
print('步骤5: 计算建设紧迫度指数')
print('=' * 60)


def entropy_weight(matrix):
    n, m = matrix.shape
    w = np.ones(m) / m
    for j in range(m):
        col = matrix[:, j]
        col_min, col_max = col.min(), col.max()
        if col_max - col_min < 1e-8:
            w[j] = 0.0; continue
        p_j = (col - col_min) / (col_max - col_min)
        p_j = np.clip(p_j, 1e-10, 1)
        e_j = -np.sum(p_j * np.log(p_j)) / np.log(n)
        w[j] = 1 - e_j
    return w / max(np.sum(w), 1e-8)


urgency_matrix = np.column_stack([
    df_gap['缺口率'].values,
    df_gap['电网负载率'].values,
    1 - df_gap['当前覆盖率'].values
])

urgency_weights = entropy_weight(urgency_matrix)
w1, w2, w3 = urgency_weights
print(f'熵权法权重: 缺口率={w1:.4f}, 电网压力={w2:.4f}, 覆盖率不足={w3:.4f}')

df_gap['紧迫度指数'] = (w1 * urgency_matrix[:, 0] +
                       w2 * urgency_matrix[:, 1] +
                       w3 * urgency_matrix[:, 2])

urgency_min, urgency_max = df_gap['紧迫度指数'].min(), df_gap['紧迫度指数'].max()
df_gap['紧迫度指数_归一化'] = ((df_gap['紧迫度指数'] - urgency_min) /
                            (urgency_max - urgency_min) * 100)
df_gap = df_gap.sort_values('紧迫度指数_归一化', ascending=False).reset_index(drop=True)

print('\n建设紧迫度排名：')
print(df_gap[['区域名称', '区域类型', '缺口率', '电网负载率',
              '当前覆盖率', '紧迫度指数_归一化']].to_string())

# =============================================================================
# 6. 空间溢出权重矩阵
# =============================================================================
print('\n' + '=' * 60)
print('步骤6: 构建空间溢出权重矩阵')
print('=' * 60)

region_coords = {
    1:  (0.0,  0.0), 2:  (2.5,  1.0), 3:  (4.0, -1.5),
    4:  (6.0,  3.0), 5:  (8.0, -0.5), 6:  (10.0, 0.0),
    7:  (-5.0, -8.0),8:  (3.0,  -7.0), 9:  (8.0,  -6.0),
    10: (2.0,  7.0),
}

n_regions = 10
dist_matrix = np.zeros((n_regions, n_regions))
for i in range(n_regions):
    for j in range(n_regions):
        xi, yi = region_coords[i + 1]; xj, yj = region_coords[j + 1]
        dist_matrix[i, j] = np.sqrt((xi - xj)**2 + (yi - yj)**2)

region_type_map = {'老城核心区': R_CORE, '城市新区': R_NEW, '城郊/工业区': R_SUBURB}

service_radii = np.array([
    region_type_map.get(df_gap.loc[df_gap['区域编号'] == i+1, '区域类型'].values[0], R_NEW)
    for i in range(n_regions)
])

spillover_matrix = np.zeros((n_regions, n_regions))
for i in range(n_regions):
    for j in range(n_regions):
        if i == j:
            spillover_matrix[i, j] = 1.0
        else:
            spillover_matrix[i, j] = np.exp(-dist_matrix[i, j] / service_radii[i])

print(f'空间溢出矩阵 (10×10):')
print(f'  最小溢出权重: {spillover_matrix[spillover_matrix < 1].min():.4f}')
print(f'  平均溢出权重(非对角): {spillover_matrix[spillover_matrix < 1].mean():.4f}')

# =============================================================================
# 7. 保存
# =============================================================================
print('\n' + '=' * 60)
print('步骤7: 保存预处理结果')
print('=' * 60)

df_table1 = df_gap[[
    '区域编号', '区域名称', '区域类型',
    '预测日均需求_MWh', '预测日均车次',
    '现有服务能力_车次日', '供需缺口_车次日', '缺口率',
    '峰值负荷_kWh', '电网总容量_kW', '电网负载率',
    '最小剩余容量_kW', '过载风险',
    '当前覆盖率', '紧迫度指数_归一化'
]].copy()
df_table1.columns = [
    '区域编号', '区域名称', '区域类型',
    '预测日均需求(MWh/日)', '预测日均车次(次/日)',
    '现有服务能力(车次/日)', '供需缺口(车次/日)', '缺口率(%)',
    '峰值负荷(kW)', '电网总容量(kW)', '电网负载率(%)',
    '最小剩余容量(kW)', '过载风险等级',
    '当前覆盖率(%)', '建设紧迫度指数(0-100)'
]
df_table1['缺口率(%)'] = df_table1['缺口率(%)'] * 100
df_table1['电网负载率(%)'] = df_table1['电网负载率(%)'] * 100
df_table1['当前覆盖率(%)'] = df_table1['当前覆盖率(%)'] * 100

df_table1.to_excel(FILE_Q2_TABLE1, index=False)
print(f'表1 已保存: {FILE_Q2_TABLE1}')

df_spillover = pd.DataFrame(spillover_matrix,
    index=[f'区域{i+1}' for i in range(n_regions)],
    columns=[f'区域{j+1}' for j in range(n_regions)])
df_spillover.to_excel(FILE_Q2_SPILLOVER)
print(f'空间溢出矩阵已保存: {FILE_Q2_SPILLOVER}')

df_dist = pd.DataFrame(dist_matrix,
    index=[f'区域{i+1}' for i in range(n_regions)],
    columns=[f'区域{j+1}' for j in range(n_regions)])
df_dist.to_excel(FILE_Q2_DISTANCE)
print(f'距离矩阵已保存: {FILE_Q2_DISTANCE}')

np.savez(FILE_Q2_PREPROCESS,
         spillover_matrix=spillover_matrix,
         dist_matrix=dist_matrix,
         service_radii=service_radii,
         urgency_weights=urgency_weights,
         region_coords=np.array(list(region_coords.values())))

print(f'预处理数据已保存至: {FILE_Q2_PREPROCESS}')
print('=' * 60)
print('problem2_data.py 运行完成！')
print('=' * 60)
