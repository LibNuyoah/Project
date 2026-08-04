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
  - output/merged_data.pkl
  - output/preprocess_data.npz

输出：
  - output/dispatch_result_uniform.npz
  - output/dispatch_result_waterfill.npz
=============================================================================
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

REGION_NAMES = ['宝塔山街道', '南市街道', '凤凰山街道', '枣园街道', '桥沟街道',
                '新城街道', '柳林镇', '河庄坪镇', '姚店镇', '李渠镇']

# 加载数据
print('=' * 60)
print('问题三 Step 2: 负荷转移计算')
print('=' * 60)

df = pd.read_pickle('output/merged_data.pkl')
data = np.load('output/preprocess_data.npz', allow_pickle=True)

PEAK_HOURS = data['peak_hours'].tolist()
FLAT_HOURS = data['flat_hours'].tolist()
VALLEY_HOURS = data['valley_hours'].tolist()
ETA = float(data['eta'])
OVERLOAD_THRESHOLD = float(data['overload_threshold'])

print(f'高峰: {PEAK_HOURS}  平段: {FLAT_HOURS}  低谷: {VALLEY_HOURS}')
print(f'转移率: {ETA*100}%')


# =============================================================================
# 方案A: 均匀分配
# =============================================================================
def apply_uniform(df):
    """均匀分配：转移总量÷低谷小时数，每低谷小时加等量"""
    df_a = df.copy()
    df_a['调度后负荷'] = df_a['充电负荷'].values

    for (rid, dtype), grp in df.groupby(['区域编号', '日期类型']):
        mask = (df_a['区域编号'] == rid) & (df_a['日期类型'] == dtype)
        idx = df_a[mask].index

        peak_idx = idx[df_a.loc[idx, '小时'].isin(PEAK_HOURS)]
        valley_idx = idx[df_a.loc[idx, '小时'].isin(VALLEY_HOURS)]

        peak_loads = df_a.loc[peak_idx, '充电负荷'].values
        valley_loads = df_a.loc[valley_idx, '充电负荷'].values

        # 转移量
        transfer_per_hour = ETA * peak_loads
        Q_total = transfer_per_hour.sum()

        # 高峰削减
        df_a.loc[peak_idx, '调度后负荷'] = peak_loads - transfer_per_hour

        # 低谷均匀填充
        df_a.loc[valley_idx, '调度后负荷'] = valley_loads + Q_total / len(valley_loads)

    return df_a


# =============================================================================
# 方案B: 填谷优先（water-filling）
# =============================================================================
def water_filling(valley_loads, Q):
    """
    将转移量Q优先填入负荷最低的时段，逐层填平。

    参数:
        valley_loads: shape (n,), 低谷各小时原始负荷
        Q: 标量，转移总量
    返回:
        filled: shape (n,), 填谷后低谷段负荷
    """
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
    """填谷优先分配"""
    df_b = df.copy()
    df_b['调度后负荷'] = df_b['充电负荷'].values

    for (rid, dtype), grp in df.groupby(['区域编号', '日期类型']):
        mask = (df_b['区域编号'] == rid) & (df_b['日期类型'] == dtype)
        idx = df_b[mask].index

        peak_idx = idx[df_b.loc[idx, '小时'].isin(PEAK_HOURS)]
        valley_idx = idx[df_b.loc[idx, '小时'].isin(VALLEY_HOURS)]

        peak_loads = df_b.loc[peak_idx, '充电负荷'].values
        valley_loads = df_b.loc[valley_idx, '充电负荷'].values

        # 转移量
        transfer_per_hour = ETA * peak_loads
        Q_total = transfer_per_hour.sum()

        # 高峰削减
        df_b.loc[peak_idx, '调度后负荷'] = peak_loads - transfer_per_hour

        # 填谷优先
        filled = water_filling(valley_loads, Q_total)
        df_b.loc[valley_idx, '调度后负荷'] = filled

    return df_b


# =============================================================================
# 执行 + 对比
# =============================================================================
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

# 对比：哪些区域/日期类型有差异
diff_count = 0
for (rid, dtype), grp in df.groupby(['区域编号', '日期类型']):
    mask_u = (df_uniform['区域编号'] == rid) & (df_uniform['日期类型'] == dtype)
    mask_w = (df_waterfill['区域编号'] == rid) & (df_waterfill['日期类型'] == dtype)
    u = df_uniform.loc[mask_u, '调度后负荷'].values
    w = df_waterfill.loc[mask_w, '调度后负荷'].values
    if not np.allclose(u, w, rtol=1e-4):
        valley_u = u[[h in VALLEY_HOURS for h in df_uniform.loc[mask_u, '小时']]]
        valley_w = w[[h in VALLEY_HOURS for h in df_waterfill.loc[mask_w, '小时']]]
        diff_count += 1
        rname = REGION_NAMES[rid-1]
        print(f'  差异: {rname} {dtype} 低谷段 max|Δ|={np.abs(valley_u-valley_w).max():.1f}kW')

if diff_count == 0:
    print('  两种方案在所有区域/日期类型上低谷分配完全一致。')
else:
    print(f'  共{diff_count}个(区域×日期类型)存在差异。')

# =============================================================================
# 保存
# =============================================================================
df_uniform.to_pickle('output/dispatch_uniform.pkl')
df_waterfill.to_pickle('output/dispatch_waterfill.pkl')

np.savez('output/dispatch_compare.npz',
         uniform_vs_waterfill_diff=diff_count)

print('\n' + '=' * 60)
print('problem3_solve.py 完成！')
print('=' * 60)
