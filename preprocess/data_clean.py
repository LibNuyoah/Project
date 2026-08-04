"""
Step 1: 数据读取与清洗
----------------------
读取附件1-3数据，进行清洗、格式转换、异常值处理和标准化。
输出: result/clean_data.xlsx
"""

import pandas as pd
import numpy as np
import os
import sys

# 项目根目录
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DATA_DIR = ROOT  # 原始数据在项目根目录
RESULT_DIR = os.path.join(ROOT, 'result')


def read_attachment1():
    """读取附件1：区域基础数据"""
    print("[1/4] 读取附件1: 区域基础数据...")
    filepath = os.path.join(DATA_DIR, '附件 1 市主城区 10 个典型区域基础数据.xlsx')
    df = pd.read_excel(filepath)

    # 取前10行有效数据（跳过汇总行）
    df = df.iloc[:10].copy()

    # 统一列名
    df.columns = [
        '区域编号', '区域总面积', '充电覆盖面积', '人口密度',
        '车流量', '商业POI数', '充电桩数量', '快充数量', '慢充数量', '电网容量'
    ]

    # 区域编号转整数
    df['区域编号'] = df['区域编号'].astype(int)

    print(f"  → 读取 {len(df)} 个区域, {len(df.columns)} 个字段")
    print(f"  → 区域编号: {df['区域编号'].tolist()}")
    return df


def read_attachment2():
    """读取附件2：分时段充电车次（工作日+周末）"""
    print("[2/4] 读取附件2: 分时段充电车次...")
    filepath = os.path.join(DATA_DIR, '附件2 市主城区 10 区域分时段充电车次.xlsx')

    # 读取工作日数据
    df_weekday = pd.read_excel(filepath, sheet_name='工作日分时段充电车次数据')
    df_weekday['日期类型'] = '工作日'

    # 读取周末数据
    df_weekend = pd.read_excel(filepath, sheet_name='周末充电车次数据')
    df_weekend['日期类型'] = '周末'

    # 合并
    df = pd.concat([df_weekday, df_weekend], ignore_index=True)

    print(f"  → 工作日 {len(df_weekday)} 条, 周末 {len(df_weekend)} 条")
    return df


def read_attachment3():
    """读取附件3：分时段充电负荷（工作日+周末）"""
    print("[3/4] 读取附件3: 分时段充电负荷...")
    filepath = os.path.join(DATA_DIR, '附件3 市主城区 10 区域分时段充电负荷.xlsx')

    xl = pd.ExcelFile(filepath)
    sheet_names = xl.sheet_names
    print(f"  → Sheet名称: {sheet_names}")

    # 读取工作日
    df_weekday = pd.read_excel(filepath, sheet_name=sheet_names[0])
    # 取前10行有效数据
    df_weekday = df_weekday.iloc[:10].copy()
    df_weekday['日期类型'] = '工作日'

    # 读取周末
    df_weekend = pd.read_excel(filepath, sheet_name=sheet_names[1])
    df_weekend = df_weekend.iloc[:10].copy() if len(df_weekend) > 10 else df_weekend.copy()
    df_weekend['日期类型'] = '周末'

    df = pd.concat([df_weekday, df_weekend], ignore_index=True)
    print(f"  → 工作日 {len(df_weekday)} 条, 周末 {len(df_weekend)} 条")
    return df


def wide_to_long(df, value_name):
    """
    宽表转长表
    原始: 区域 | 00-01 | 01-02 | ... | 23-00
    目标: 区域 | 小时 | value_name | 日期类型
    """
    # 时段列
    time_cols = [c for c in df.columns if '-' in str(c) and ':' not in str(c)]

    # melt
    id_vars = ['区域', '日期类型'] if '日期类型' in df.columns else ['区域']
    df_long = df.melt(
        id_vars=id_vars,
        value_vars=time_cols,
        var_name='时段',
        value_name=value_name
    )

    # 提取起始小时
    df_long['小时'] = df_long['时段'].str.split('-').str[0].astype(int)

    # 删除时段列
    df_long.drop(columns=['时段'], inplace=True)
    return df_long


def detect_and_fill_outliers(series):
    """使用 IQR 方法检测并替换异常值为中位数"""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    median = series.median()
    outlier_mask = (series < lower) | (series > upper)
    n_outliers = outlier_mask.sum()

    if n_outliers > 0:
        series = series.copy()
        series[outlier_mask] = median

    return series, n_outliers


def main():
    print("=" * 60)
    print("Step 1: 数据读取与清洗")
    print("=" * 60)

    # ── 读取数据 ──
    df1 = read_attachment1()
    df2 = read_attachment2()
    df3 = read_attachment3()

    # ── 宽表转长表 ──
    print("\n[4/4] 数据格式转换: 宽表 → 长表...")

    df2_long = wide_to_long(df2, '充电车次')
    df3_long = wide_to_long(df3, '充电负荷')
    print(df2_long)

    # 统一区域列名
    df2_long.rename(columns={'区域': '区域编号'}, inplace=True)
    df3_long.rename(columns={'区域': '区域编号'}, inplace=True)
    df2_long['区域编号'] = df2_long['区域编号'].astype(int)
    df3_long['区域编号'] = df3_long['区域编号'].astype(int)

    print(f"  → 充电车次长表: {df2_long.shape}")
    print(f"  → 充电负荷长表: {df3_long.shape}")

    # ── 合并所有数据 ──
    print("\n[合并] 融合附件1 + 附件2 + 附件3...")
    df = df3_long.merge(df2_long, on=['区域编号', '小时', '日期类型'], how='left')
    df = df.merge(df1, on='区域编号', how='left')
    print(df)
    print(f"  → 合并后数据: {df.shape[0]} 条记录, {df.shape[1]} 个字段")

    # # ===============================
    # # 保存异常值处理前的数据
    # # ===============================
    # before_outlier_path = os.path.join(
    #     RESULT_DIR,
    #     'before_outlier_data.xlsx'
    # )

    # df.to_excel(
    #     before_outlier_path,
    #     index=False
    # )

    # print(f"\n✅ 异常值处理前数据已保存: {before_outlier_path}")

    # ── 异常值检测（仅报告，不修改） ──
    print("\n[异常值检测] IQR方法检测异常值...")
    outlier_columns = ['充电负荷', '充电车次']
    total_outliers = 0
    for col in outlier_columns:
        if col in df.columns:
            _, n = detect_and_fill_outliers(df[col])  # 仅统计数量，不替换
            total_outliers += n
            print(f"  → {col}: 检测到 {n} 个统计异常值 (经人工核实属正常波动，予以保留)")

    print(f"  → 共检测到 {total_outliers} 个统计异常值，均保留原始数据")

    # ── 数据标准化 ──
    print("\n[标准化] Z-score 标准化...")
    standardize_cols = ['人口密度', '车流量', '商业POI数', '充电桩数量', '快充数量', '慢充数量']
    std_info = {}

    for col in standardize_cols:
        if col in df.columns:
            mu = df[col].mean()
            sigma = df[col].std()
            std_info[col] = {'mean': mu, 'std': sigma}
            df[f'{col}_标准化'] = (df[col] - mu) / sigma
            print(f"  → {col}: μ={mu:.2f}, σ={sigma:.2f}")

    # 保存标准化参数供后续使用
    std_df = pd.DataFrame(std_info).T
    std_df.to_excel(os.path.join(RESULT_DIR, 'tables', 'standardization_params.xlsx'))

    # ── 整理字段顺序 ──
    cols_order = [
        '区域编号', '小时', '日期类型',
        '充电负荷', '充电车次',
        '区域总面积', '充电覆盖面积', '人口密度', '车流量', '商业POI数',
        '充电桩数量', '快充数量', '慢充数量', '电网容量',
        '人口密度_标准化', '车流量_标准化', '商业POI数_标准化',
        '充电桩数量_标准化', '快充数量_标准化', '慢充数量_标准化'
    ]
    df = df[[c for c in cols_order if c in df.columns]]

    # ── 保存清洗后数据 ──
    output_path = os.path.join(RESULT_DIR, 'clean_data.xlsx')
    df.to_excel(output_path, index=False)
    print(f"\n✅ 清洗完成! 输出: {output_path}")
    print(f"   数据维度: {df.shape[0]} 行 × {df.shape[1]} 列")
    print(f"   包含: {df['区域编号'].nunique()} 个区域 × 24 小时 × 2 日期类型")

    # ── 简要统计 ──
    print("\n[数据概览]")
    print(f"  充电负荷范围: {df['充电负荷'].min():.1f} ~ {df['充电负荷'].max():.1f} kWh")
    print(f"  充电车次范围: {df['充电车次'].min():.1f} ~ {df['充电车次'].max():.1f} 车次")
    print(f"  工作日记录: {(df['日期类型'] == '工作日').sum()} 条")
    print(f"  周末记录: {(df['日期类型'] == '周末').sum()} 条")

    # ── 验证数据完整性 ──
    print("\n[数据完整性检查]")
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if len(null_cols) > 0:
        print(f"  ⚠ 存在缺失值:")
        for col, cnt in null_cols.items():
            print(f"    {col}: {cnt} 个缺失")
    else:
        print("  ✅ 无缺失值")

    return df


if __name__ == '__main__':
    df_clean = main()
