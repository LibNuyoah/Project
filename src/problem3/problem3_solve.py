"""
=============================================================================
problem3_solve.py — 问题三：负荷转移计算（方案A均匀 vs 方案B填谷优先）
=============================================================================
功能：
  1. 加载预处理数据
  2. 方案A：均匀分配到低谷各小时
  3. 方案B：water-filling 填谷优先
  4. 输出两种方案的调度后负荷

输入：
  - results/q3_output/merged_data.pkl
  - results/q3_output/preprocess_data.npz

输出：
  - results/q3_output/dispatch_uniform.pkl
  - results/q3_output/dispatch_waterfill.pkl
  - results/q3_output/dispatch_compare.npz
=============================================================================
"""

import pandas as pd
import numpy as np
import os, sys
import warnings
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from utils.paths import (
    RESULTS_Q3, FILE_Q3_MERGED_DATA, FILE_Q3_PREPROCESS,
    FILE_Q3_DISPATCH_UNIFORM, FILE_Q3_DISPATCH_WATERFILL, FILE_Q3_DISPATCH_COMPARE
)

REGION_NAMES = ['宝塔山街道', '南市街道', '凤凰山街道', '枣园街道', '桥沟街道',
                '新城街道', '柳林镇', '河庄坪镇', '姚店镇', '李渠镇']

# 加载数据
print('=' * 60)
print('问题三 Step 2: 负荷转移计算')
print('=' * 60)

df = pd.read_pickle(FILE_Q3_MERGED_DATA)
data = np.load(FILE_Q3_PREPROCESS, allow_pickle=True)

PEAK_HOURS = data['peak_hours'].tolist()
FLAT_HOURS = data['flat_hours'].tolist()
VALLEY_HOURS = data['valley_hours'].tolist()
ETA = float(data['eta'])
OVERLOAD_THRESHOLD = float(data['overload_threshold'])

print(f'高峰: {PEAK_HOURS}  平段: {FLAT_HOURS}  低谷: {VALLEY_HOURS}')
print(f'转移率: {ETA*100}%')


def apply_uniform(df):
    df_a = df.copy()
    df_a['调度后负荷'] = df_a['充电负荷'].astype(float).values
    for (rid, dtype), grp in df.groupby(['区域编号', '日期类型']):
        mask = (df_a['区域编号'] == rid) & (df_a['日期类型'] == dtype)
        idx = df_a[mask].index
        peak_idx = idx[df_a.loc[idx, '小时'].isin(PEAK_HOURS)]
        valley_idx = idx[df_a.loc[idx, '小时'].isin(VALLEY_HOURS)]
        peak_loads = df_a.loc[peak_idx, '充电负荷'].values
        valley_loads = df_a.loc[valley_idx, '充电负荷'].values
        transfer_per_hour = ETA * peak_loads
        Q_total = transfer_per_hour.sum()
        df_a.loc[peak_idx, '调度后负荷'] = peak_loads - transfer_per_hour
        df_a.loc[valley_idx, '调度后负荷'] = valley_loads + Q_total / len(valley_loads)
    return df_a


def water_filling(valley_loads, Q):
    n = len(valley_loads)
    loads = valley_loads.copy().astype(float)
    order = np.argsort(loads)
    sorted_loads = loads[order]
    remaining = Q
    for k in range(n - 1):
        gap = sorted_loads[k + 1] - sorted_loads[k]
        needed = gap * (k + 1)
        if remaining >= needed:
            sorted_loads[:k + 1] += gap
            remaining -= needed
        else:
            sorted_loads[:k + 1] += remaining / (k + 1)
            remaining = 0
            break
    if remaining > 0:
        sorted_loads += remaining / n
    result = np.zeros(n)
    for i, idx in enumerate(order):
        result[idx] = sorted_loads[i]
    return result


def apply_water_filling(df):
    df_b = df.copy()
    df_b['调度后负荷'] = df_b['充电负荷'].astype(float).values
    for (rid, dtype), grp in df.groupby(['区域编号', '日期类型']):
        mask = (df_b['区域编号'] == rid) & (df_b['日期类型'] == dtype)
        idx = df_b[mask].index
        peak_idx = idx[df_b.loc[idx, '小时'].isin(PEAK_HOURS)]
        valley_idx = idx[df_b.loc[idx, '小时'].isin(VALLEY_HOURS)]
        peak_loads = df_b.loc[peak_idx, '充电负荷'].values
        valley_loads = df_b.loc[valley_idx, '充电负荷'].values
        transfer_per_hour = ETA * peak_loads
        Q_total = transfer_per_hour.sum()
        df_b.loc[peak_idx, '调度后负荷'] = peak_loads - transfer_per_hour
        filled = water_filling(valley_loads, Q_total)
        df_b.loc[valley_idx, '调度后负荷'] = filled
    return df_b


# 执行
print('\n方案A: 均匀分配...')
df_uniform = apply_uniform(df)

print('方案B: 填谷优先...')
df_waterfill = apply_water_filling(df)

# 快速对比
print('\n' + '=' * 60)
print('方案A vs 方案B 快速对比（全市工作日）')
print('=' * 60)
wd = df['日期类型'] == '工作日'
for label, df_a in [('A 均匀', df_uniform), ('B 填谷', df_waterfill)]:
    city_after = df_a[wd].groupby('小时')['调度后负荷'].sum()
    pk, vy = city_after.max(), city_after.min()
    print(f'  方案{label}: 峰值={pk:.0f}kW, 谷值={vy:.0f}kW, 峰谷差={pk-vy:.0f}kW')

# 保存
df_uniform.to_pickle(FILE_Q3_DISPATCH_UNIFORM)
df_waterfill.to_pickle(FILE_Q3_DISPATCH_WATERFILL)
np.savez(FILE_Q3_DISPATCH_COMPARE, uniform_vs_waterfill_diff=0)

print('\n' + '=' * 60)
print('problem3_solve.py 完成！')
print('=' * 60)
