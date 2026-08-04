"""
=============================================================================
problem3_data.py — 问题三：数据加载与调度前分析
=============================================================================
功能：
  1. 读取附件3（充电负荷）+ 附件4（电网容量上限）
  2. 宽表→长表转换、数据合并
  3. 调度前峰谷差、负荷率、过载风险计算
  4. 输出 表_调度前分析.xlsx

输入：
  - 附件3 市主城区 10 区域分时段充电负荷.xlsx
  - 附件4 市主城区 10 个区域分时段电网最大允许负荷数据.xlsx

输出：
  - output/表_调度前分析.xlsx
  - output/preprocess_data.npz
=============================================================================
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('output', exist_ok=True)

REGION_NAMES = ['宝塔山街道', '南市街道', '凤凰山街道', '枣园街道', '桥沟街道',
                '新城街道', '柳林镇', '河庄坪镇', '姚店镇', '李渠镇']
N_REGIONS = 10

# =============================================================================
# 0. 附件5参数（硬编码）
# =============================================================================
PEAK_HOURS   = list(range(11, 14)) + list(range(16, 23))
FLAT_HOURS   = list(range(7, 11)) + list(range(14, 16))
VALLEY_HOURS = list(range(0, 7))

ETA = 0.20
OVERLOAD_THRESHOLD = 2100  # kW
PRICE_PEAK   = 1.19
PRICE_FLAT   = 0.70
PRICE_VALLEY = 0.21

print('=' * 60)
print('问题三 Step 1: 数据加载与调度前分析')
print('=' * 60)
print(f'高峰: {PEAK_HOURS} ({len(PEAK_HOURS)}h)  平段: {FLAT_HOURS} ({len(FLAT_HOURS)}h)  低谷: {VALLEY_HOURS} ({len(VALLEY_HOURS)}h)')
print(f'转移率: {ETA*100}%  过载阈值: {OVERLOAD_THRESHOLD}kW')

# =============================================================================
# 1. 读取附件3：分时段充电负荷（工作日+周末）
# =============================================================================
print('\n[1/4] 读取附件3...')
xl3 = pd.ExcelFile('../data/附件3 市主城区 10 区域分时段充电负荷.xlsx')
sheet_names = xl3.sheet_names
print(f'  Sheets: {sheet_names}')

df3_wd = pd.read_excel('../data/附件3 市主城区 10 区域分时段充电负荷.xlsx', sheet_name=sheet_names[0])
df3_wd = df3_wd.iloc[:10].copy()
df3_wd['日期类型'] = '工作日'

df3_we = pd.read_excel('../data/附件3 市主城区 10 区域分时段充电负荷.xlsx', sheet_name=sheet_names[1])
df3_we = df3_we.iloc[:10].copy()
df3_we['日期类型'] = '周末'

df3_all = pd.concat([df3_wd, df3_we], ignore_index=True)

# 统一区域列名
region_col3 = [c for c in df3_all.columns if '区域' in str(c)][0]
df3_all.rename(columns={region_col3: '区域编号'}, inplace=True)
df3_all['区域编号'] = df3_all['区域编号'].astype(int)

# 宽→长
time_cols3 = [c for c in df3_all.columns if '-' in str(c)]
df3_long = df3_all.melt(
    id_vars=['区域编号', '日期类型'],
    value_vars=time_cols3, var_name='时段', value_name='充电负荷'
)
df3_long['小时'] = df3_long['时段'].str.split('-').str[0].astype(int)
df3_long.drop(columns=['时段'], inplace=True)
print(f'  附件3长表: {df3_long.shape} (区域×小时×日期类型)')

# =============================================================================
# 2. 读取附件4：电网最大允许负荷
# =============================================================================
print('\n[2/4] 读取附件4...')
df4_raw = pd.read_excel('../data/附件4 市主城区 10 个区域分时段电网最大允许负荷数据.xlsx')
df4_raw = df4_raw.iloc[:10].copy()
time_cols4 = [c for c in df4_raw.columns if '-' in str(c)]

records4 = []
for i in range(10):
    rid = int(df4_raw.iloc[i, 1])
    for h, tc in enumerate(time_cols4):
        records4.append({
            '区域编号': rid,
            '小时': h,
            '电网允许负荷': float(df4_raw.iloc[i][tc])
        })
df4_long = pd.DataFrame(records4)
print(f'  附件4长表: {df4_long.shape}')

# =============================================================================
# 3. 合并数据
# =============================================================================
print('\n[3/4] 合并数据...')
df = df3_long.merge(df4_long, on=['区域编号', '小时'], how='left')
print(f'  合并后: {df.shape} ({df["区域编号"].nunique()}区×{df["小时"].nunique()}h×{df["日期类型"].nunique()}类型)')

# 验证
assert df.shape[0] == 480, f"期望480行，实际{df.shape[0]}"
assert df.isnull().sum().sum() == 0, "存在缺失值"

# =============================================================================
# 4. 调度前分析
# =============================================================================
print('\n[4/4] 调度前分析...')


def compute_metrics(load_series, grid_cap_series):
    """单区域×日期类型指标计算"""
    peak, valley, mean = load_series.max(), load_series.min(), load_series.mean()
    return {
        'peak': peak, 'valley': valley, 'mean': mean,
        'delta_p': peak - valley,
        'ratio': peak / (valley + 1.0),
        'load_rate': mean / peak * 100 if peak > 0 else 0,
        'overload_annex4': int((load_series > grid_cap_series).sum()),
        'overload_2100': int((load_series > OVERLOAD_THRESHOLD).sum()),
    }


results = []
for (rid, dtype), grp in df.groupby(['区域编号', '日期类型']):
    m = compute_metrics(grp['充电负荷'].values, grp['电网允许负荷'].values)
    m['区域编号'] = rid
    m['日期类型'] = dtype
    results.append(m)

df_before = pd.DataFrame(results)
df_before['区域名称'] = df_before['区域编号'].apply(lambda x: REGION_NAMES[x-1])

# 全市汇总
city_rows = []
for dtype in ['工作日', '周末']:
    mask = df['日期类型'] == dtype
    city_load = df[mask].groupby('小时')['充电负荷'].sum()
    city_grid = df[mask].groupby('小时')['电网允许负荷'].sum()
    m = compute_metrics(city_load.values, city_grid.values)
    m['区域编号'] = 0
    m['区域名称'] = '全市汇总'
    m['日期类型'] = dtype
    city_rows.append(m)

df_before_all = pd.concat([df_before, pd.DataFrame(city_rows)], ignore_index=True)

# 打印关键结果
for dtype in ['工作日', '周末']:
    sub = df_before[df_before['日期类型'] == dtype]
    city = df_before_all[(df_before_all['区域编号'] == 0) & (df_before_all['日期类型'] == dtype)].iloc[0]
    print(f'\n  [{dtype}]')
    print(f'  全市: 峰值={city["peak"]:.0f}kW, 谷值={city["valley"]:.0f}kW, '
          f'峰谷差={city["delta_p"]:.0f}kW, 负荷率={city["load_rate"]:.1f}%')
    print(f'  超2100kW区域: {sum(sub["overload_2100"] > 0)}个')

# =============================================================================
# 5. 保存
# =============================================================================
output_cols = ['区域编号', '区域名称', '日期类型',
               'peak', 'valley', 'delta_p', 'ratio', 'load_rate',
               'overload_annex4', 'overload_2100']
df_table = df_before_all[output_cols].copy()
df_table.rename(columns={
    'peak': '峰值负荷(kW)', 'valley': '谷值负荷(kW)',
    'delta_p': '峰谷差(kW)', 'ratio': '峰谷比',
    'load_rate': '负荷率(%)', 'overload_annex4': '超附件4上限时段数',
    'overload_2100': '超2100kW时段数'
}, inplace=True)

path_table = 'output/表_调度前分析.xlsx'
df_table.to_excel(path_table, index=False)
print(f'\n表已保存: {path_table}')

# 保存核心数据供后续使用
np.savez('output/preprocess_data.npz',
         peak_hours=PEAK_HOURS, flat_hours=FLAT_HOURS, valley_hours=VALLEY_HOURS,
         eta=ETA, overload_threshold=OVERLOAD_THRESHOLD,
         price_peak=PRICE_PEAK, price_flat=PRICE_FLAT, price_valley=PRICE_VALLEY)

# 保存完整合并数据
df.to_pickle('output/merged_data.pkl')

print('\n' + '=' * 60)
print('problem3_data.py 完成！')
print('=' * 60)
