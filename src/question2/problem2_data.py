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
  - ../附件 1 市主城区 10 个典型区域基础数据.xlsx
  - ../附件4 市主城区 10 个区域分时段电网最大允许负荷数据.xlsx
  - ../prediction_result.xlsx          (问题1输出)
  - ../hourly_prediction.xlsx          (问题1输出)

输出文件：
  - output/表1_各区域供需缺口与建设紧迫度.xlsx
  - output/空间溢出权重矩阵.xlsx
=============================================================================
"""

import pandas as pd
import numpy as np
import os, sys
import warnings
warnings.filterwarnings('ignore')

# 项目根目录 (problem2_data.py 在 src/question2/ 下，向上3级)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
OUTPUT_DIR = os.path.join(ROOT, 'result', 'tables')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# 0. 创建输出目录
# =============================================================================
os.makedirs('output', exist_ok=True)

# =============================================================================
# 1. 加载原始数据
# =============================================================================
print('=' * 60)
print('步骤1: 加载原始数据')
print('=' * 60)

# 附件1：区域基础数据（取前10行有效数据，跳过汇总行）
df_annex1 = pd.read_excel(os.path.join(ROOT, 'data/raw/附件 1 市主城区 10 个典型区域基础数据.xlsx'), nrows=10)
print(f'附件1: {df_annex1.shape[0]} 个区域, {df_annex1.shape[1]} 个字段')

# 附件4：24小时电网最大允许负荷
df_annex4 = pd.read_excel(os.path.join(ROOT, 'data/raw/附件4 市主城区 10 个区域分时段电网最大允许负荷数据.xlsx'))
print(f'附件4: {df_annex4.shape[0]} 个区域, {df_annex4.shape[1]} 列(含区域名+24h)')

# 问题1输出：日均预测结果
df_pred_daily = pd.read_excel(os.path.join(ROOT, 'result/prediction_result.xlsx'))
print(f'prediction_result: {df_pred_daily.shape[0]} 个区域')

# 问题1输出：分时预测结果 (480行 = 10区域 × 24h × 2日期类型)
df_pred_hourly = pd.read_excel(os.path.join(ROOT, 'result/tables/hourly_prediction.xlsx'))
print(f'hourly_prediction: {df_pred_hourly.shape[0]} 条记录')

# =============================================================================
# 2. 附件5参数（硬编码，便于修改）
# =============================================================================
print('\n' + '=' * 60)
print('步骤2: 加载附件5参数')
print('=' * 60)

# 充电桩参数
COST_FAST = 6.0        # 快充桩单台建设成本（万元）
COST_SLOW = 0.8        # 慢充桩单台建设成本（万元）
CAP_FAST  = 80         # 快充桩日服务能力（车次/日）
CAP_SLOW  = 20         # 慢充桩日服务能力（车次/日）
POWER_FAST = 120       # 快充桩功率（kW）
POWER_SLOW = 7         # 慢充桩功率（kW）

# 服务半径（km）
R_CORE  = 1.5          # 核心城区
R_NEW   = 2.0          # 文旅区（附件5写文旅区2km，这里统一作为中距离）
R_SUBURB = 2.5         # 城郊区

# 覆盖率约束
COVERAGE_MIN = 0.90    # 最低服务覆盖率 90%

# 电网过载判定
OVERLOAD_THRESHOLD = 2100  # kW
OVERLOAD_DURATION  = 15    # 分钟（简化处理，此处以小时级判断）

# 充电同时率（新增桩对峰值负荷的贡献系数）
SIMULTANEITY = 0.8

# 单车平均充电量（从附件3工作日/周末数据反推）
# 工作日: 总负荷中位数约569 kWh/h / 总车次约44车次/h ≈ 约13 kWh/车次
# 此处简化，取12 kWh作为单车平均充电估算
AVG_CHARGE_PER_CAR = 12.0  # kWh/车次

print(f'快充桩: {COST_FAST}万元/台, {CAP_FAST}车次/日, {POWER_FAST}kW')
print(f'慢充桩: {COST_SLOW}万元/台, {CAP_SLOW}车次/日, {POWER_SLOW}kW')
print(f'服务半径: 核心{R_CORE}km / 文旅{R_NEW}km / 城郊{R_SUBURB}km')
print(f'最低覆盖率: {COVERAGE_MIN*100}%')
print(f'单车平均充电: {AVG_CHARGE_PER_CAR} kWh')

# =============================================================================
# 3. 各区域供需缺口计算
# =============================================================================
print('\n' + '=' * 60)
print('步骤3: 计算各区域充电供需缺口')
print('=' * 60)

# 从prediction_result提取关键字段
df_gap = df_pred_daily[['区域编号', '区域名称', '区域类型',
                         '预测日均需求_kWh', '预测日均需求_MWh',
                         '工作日日均需求_kWh', '周末日均需求_kWh',
                         '峰值负荷_kWh']].copy()

# 从附件1提取现有桩数、面积、电网容量
df_annex1_select = df_annex1[['区域编号', '区域总面积(km²)', '充电覆盖面积面积(km²)',
                               '现有充电桩数量（个）', '其中快充桩（个）', '其中慢充桩（个）',
                               '区域电网总容量（万千瓦）']].copy()
df_annex1_select.columns = ['区域编号', '区域总面积_km2', '充电覆盖面积_km2',
                             '现有桩总数', '现有快充桩', '现有慢充桩',
                             '电网总容量_万kW']

# 合并
df_gap = df_gap.merge(df_annex1_select, on='区域编号', how='left')

# 计算现有服务能力（车次/日）
df_gap['现有服务能力_车次日'] = (df_gap['现有快充桩'] * CAP_FAST +
                               df_gap['现有慢充桩'] * CAP_SLOW)

# 计算预测日均车次需求
df_gap['预测日均车次'] = df_gap['预测日均需求_kWh'] / AVG_CHARGE_PER_CAR

# 计算供需缺口（车次/日，非负）
df_gap['供需缺口_车次日'] = np.maximum(0, df_gap['预测日均车次'] - df_gap['现有服务能力_车次日'])

# 计算缺口率
df_gap['缺口率'] = df_gap['供需缺口_车次日'] / df_gap['现有服务能力_车次日']

# 当前覆盖率（充电覆盖面积/区域总面积）
df_gap['当前覆盖率'] = df_gap['充电覆盖面积_km2'] / df_gap['区域总面积_km2']

# 电网总容量转kW
df_gap['电网总容量_kW'] = df_gap['电网总容量_万kW'] * 10000

# 电网负载率（峰值负荷/电网容量）
df_gap['电网负载率'] = df_gap['峰值负荷_kWh'] / df_gap['电网总容量_kW']

print('各区域供需缺口概况：')
print(df_gap[['区域名称', '现有服务能力_车次日', '预测日均车次',
              '供需缺口_车次日', '缺口率', '当前覆盖率', '电网负载率']].to_string())

# =============================================================================
# 4. 电网剩余容量计算（取24小时中最小剩余容量）
# =============================================================================
print('\n' + '=' * 60)
print('步骤4: 计算电网剩余容量')
print('=' * 60)

# 从hourly_prediction计算各区域工作日/周末的24小时预测负荷
df_hourly = df_pred_hourly.copy()

# 附件4的24小时电网最大允许负荷（kW）
# 先提取区域编号对应的电网容量数据
hours = [f'{h:02d}-{h+1:02d}' for h in range(24)]
hour_cols_annex4 = [c for c in df_annex4.columns if '-' in str(c)]

# 构建电网允许负荷长表
grid_capacity_records = []
for _, row in df_annex4.iterrows():
    region_id = int(row['区域']) if '区域' in df_annex4.columns else int(row.iloc[1])
    for h_idx, h_col in enumerate(hour_cols_annex4):
        grid_capacity_records.append({
            '区域编号': region_id,
            '小时': h_idx,
            '电网允许负荷_kW': row[h_col]
        })
df_grid = pd.DataFrame(grid_capacity_records)
df_grid['区域编号'] = df_grid['区域编号'].astype(int)

# 合并预测负荷与电网允许负荷
df_compare = df_hourly.merge(df_grid, on=['区域编号', '小时'], how='left')

# 计算每小时剩余容量
df_compare['剩余容量_kW'] = df_compare['电网允许负荷_kW'] - df_compare['预测负荷']

# 各区域最小剩余容量（最紧张时段）
df_min_remain = df_compare.groupby('区域编号')['剩余容量_kW'].min().reset_index()
df_min_remain.columns = ['区域编号', '最小剩余容量_kW']

# 加入峰值负荷时段的剩余容量（最危险时段）
df_peak_remain = df_compare.loc[df_compare.groupby('区域编号')['预测负荷'].idxmax(),
                                 ['区域编号', '小时', '预测负荷', '电网允许负荷_kW', '剩余容量_kW']]
df_peak_remain.columns = ['区域编号', '峰值时段', '峰值预测负荷_kW',
                           '峰值时段电网允许_kW', '峰值时段剩余容量_kW']

# 合并
df_gap = df_gap.merge(df_min_remain, on='区域编号', how='left')
df_gap = df_gap.merge(df_peak_remain[['区域编号', '峰值时段', '峰值时段剩余容量_kW']],
                       on='区域编号', how='left')

# 判断过载风险
df_gap['过载风险'] = df_gap['最小剩余容量_kW'].apply(
    lambda x: '高风险' if x < -200 else ('中风险' if x < 0 else ('低风险' if x < 500 else '安全'))
)

print('各区域电网剩余容量与过载风险：')
print(df_gap[['区域名称', '峰值负荷_kWh', '电网总容量_kW',
              '最小剩余容量_kW', '峰值时段', '过载风险']].to_string())

# =============================================================================
# 5. 建设紧迫度指数（熵权法确定权重）
# =============================================================================
print('\n' + '=' * 60)
print('步骤5: 计算建设紧迫度指数')
print('=' * 60)


def entropy_weight(matrix):
    """
    熵权法计算指标权重（带零方差保护）

    参数:
        matrix: numpy array, shape (n_samples, n_indicators)
               所有指标均为正向（越大越紧迫）

    返回:
        weights: 各指标权重（和为1，均为正）
    """
    n, m = matrix.shape
    w = np.ones(m) / m  # 默认等权重

    for j in range(m):
        col = matrix[:, j]
        col_min, col_max = col.min(), col.max()
        if col_max - col_min < 1e-8:
            # 零方差：该指标无区分度，权重置0
            w[j] = 0.0
            continue
        # Min-Max归一化
        p_j = (col - col_min) / (col_max - col_min)
        p_j = np.clip(p_j, 1e-10, 1)
        # 计算熵
        e_j = -np.sum(p_j * np.log(p_j)) / np.log(n)
        w[j] = 1 - e_j  # 差异性越大，权重越大

    # 归一化权重（排除已置零的维度）
    if np.sum(w) > 1e-8:
        w = w / np.sum(w)
    else:
        w = np.ones(m) / m
    return w


# 构建紧迫度评价矩阵（3个指标，均为正向：越大越紧迫）
urgency_matrix = np.column_stack([
    df_gap['缺口率'].values,           # 指标1: 需求缺口率
    df_gap['电网负载率'].values,        # 指标2: 电网压力
    1 - df_gap['当前覆盖率'].values     # 指标3: 覆盖率不足程度
])

# 熵权法确定权重
urgency_weights = entropy_weight(urgency_matrix)
w1, w2, w3 = urgency_weights
print(f'熵权法权重: 缺口率={w1:.4f}, 电网压力={w2:.4f}, 覆盖率不足={w3:.4f}')

# 计算紧迫度指数（加权和，归一化到0-100）
df_gap['紧迫度指数'] = (w1 * urgency_matrix[:, 0] +
                       w2 * urgency_matrix[:, 1] +
                       w3 * urgency_matrix[:, 2])

# 归一化紧迫度到0-100
urgency_min, urgency_max = df_gap['紧迫度指数'].min(), df_gap['紧迫度指数'].max()
df_gap['紧迫度指数_归一化'] = ((df_gap['紧迫度指数'] - urgency_min) /
                            (urgency_max - urgency_min) * 100)

# 按紧迫度排序
df_gap = df_gap.sort_values('紧迫度指数_归一化', ascending=False).reset_index(drop=True)

print('\n建设紧迫度排名：')
print(df_gap[['区域名称', '区域类型', '缺口率', '电网负载率',
              '当前覆盖率', '紧迫度指数_归一化']].to_string())

# =============================================================================
# 6. 区域间距离矩阵与空间溢出权重矩阵
# =============================================================================
print('\n' + '=' * 60)
print('步骤6: 构建空间溢出权重矩阵')
print('=' * 60)

# 6.1 估算10个区域的相对坐标
# ---------------------------------------------------------------------------
# 基于延安市主城区实际地理布局，按区域类型和面积估算中心坐标（km）
# 老城核心区(1,2,3,4)挤在一起，新区(5,6)在东侧，城郊(7,8,9,10)在外围
# ---------------------------------------------------------------------------
region_coords = {
    1:  (0.0,  0.0),    # 宝塔山街道 — 老城核心，坐标原点
    2:  (2.5,  1.0),    # 南市街道 — 老城核心，城南
    3:  (4.0, -1.5),    # 凤凰山街道 — 新区，城东偏南
    4:  (6.0,  3.0),    # 枣园街道 — 老城核心西北（面积大110km²）
    5:  (8.0, -0.5),    # 桥沟街道 — 新区（面积大80km²）
    6:  (10.0, 0.0),    # 新城街道 — 新区
    7:  (-5.0, -8.0),   # 柳林镇 — 城郊/工业区（面积大140km²）
    8:  (3.0,  -7.0),   # 河庄坪镇 — 城郊/工业区（面积大120km²）
    9:  (8.0,  -6.0),   # 姚店镇 — 城郊/工业区（面积大131km²）
    10: (2.0,  7.0),    # 李渠镇 — 城郊/工业区（面积小22km²）
}

# 6.2 构建距离矩阵
n_regions = 10
dist_matrix = np.zeros((n_regions, n_regions))
for i in range(n_regions):
    for j in range(n_regions):
        xi, yi = region_coords[i + 1]
        xj, yj = region_coords[j + 1]
        dist_matrix[i, j] = np.sqrt((xi - xj)**2 + (yi - yj)**2)

# 6.3 各区域服务半径映射
region_type_map = {
    '老城核心区': R_CORE,
    '城市新区': R_NEW,
    '城郊/工业区': R_SUBURB,
}

# 构建服务半径数组
service_radii = np.array([
    region_type_map.get(df_gap.loc[df_gap['区域编号'] == i+1, '区域类型'].values[0], R_NEW)
    for i in range(n_regions)
])

# 6.4 构建空间溢出权重矩阵 W_ij: 区域i的桩对区域j的服务贡献
spillover_matrix = np.zeros((n_regions, n_regions))
for i in range(n_regions):
    for j in range(n_regions):
        if i == j:
            spillover_matrix[i, j] = 1.0  # 本区域完全服务
        else:
            # 指数衰减：距离越远，溢出越弱
            spillover_matrix[i, j] = np.exp(-dist_matrix[i, j] / service_radii[i])

print(f'空间溢出矩阵 (10×10):')
print(f'  对角线=1.0, 非对角线用exp(-d/R)衰减')
print(f'  服务半径: 核心{R_CORE}km / 新区{R_NEW}km / 城郊{R_SUBURB}km')
print(f'  最小溢出权重: {spillover_matrix[spillover_matrix < 1].min():.4f}')
print(f'  平均溢出权重(非对角): {spillover_matrix[spillover_matrix < 1].mean():.4f}')

# =============================================================================
# 7. 保存预处理结果
# =============================================================================
print('\n' + '=' * 60)
print('步骤7: 保存预处理结果')
print('=' * 60)

# 表1: 各区域供需缺口与建设紧迫度
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
# 格式化为百分比
df_table1['缺口率(%)'] = df_table1['缺口率(%)'] * 100
df_table1['电网负载率(%)'] = df_table1['电网负载率(%)'] * 100
df_table1['当前覆盖率(%)'] = df_table1['当前覆盖率(%)'] * 100

output_path1 = os.path.join(OUTPUT_DIR, '表1_各区域供需缺口与建设紧迫度.xlsx')
df_table1.to_excel(output_path1, index=False)
print(f'表1 已保存: {output_path1}')

# 空间溢出权重矩阵
df_spillover = pd.DataFrame(
    spillover_matrix,
    index=[f'区域{i+1}' for i in range(n_regions)],
    columns=[f'区域{j+1}' for j in range(n_regions)]
)
output_path2 = os.path.join(OUTPUT_DIR, '空间溢出权重矩阵.xlsx')
df_spillover.to_excel(output_path2)
print(f'空间溢出矩阵已保存: {output_path2}')

# 距离矩阵
df_dist = pd.DataFrame(
    dist_matrix,
    index=[f'区域{i+1}' for i in range(n_regions)],
    columns=[f'区域{j+1}' for j in range(n_regions)]
)
output_path3 = os.path.join(OUTPUT_DIR, '区域距离矩阵.xlsx')
df_dist.to_excel(output_path3)
print(f'距离矩阵已保存: {output_path3}')

# 保存核心数据供后续步骤使用
np.savez(os.path.join(OUTPUT_DIR, 'preprocess_data.npz'),
         spillover_matrix=spillover_matrix,
         dist_matrix=dist_matrix,
         service_radii=service_radii,
         urgency_weights=urgency_weights,
         region_coords=np.array(list(region_coords.values())))

print('\n预处理数据已保存至 output/preprocess_data.npz')
print('=' * 60)
print('problem2_data.py 运行完成！')
print('=' * 60)
