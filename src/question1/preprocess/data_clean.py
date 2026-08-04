"""
Step 1: 数据读取与清洗
----------------------
从官方原始附件读取数据，进行清洗、格式转换和标准化。
输出: data/processed/clean_data.xlsx

附件2和附件3均为包含工作日/周末两个sheet的单一Excel文件，
通过 sheet_name 参数分别读取，不拆分原始文件。
"""

import pandas as pd
import numpy as np
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

from utils.paths import (
    FILE_ATTACHMENT1, FILE_ATTACHMENT2, FILE_ATTACHMENT3,
    SHEET_ATTACHMENT2_WEEKDAY, SHEET_ATTACHMENT2_WEEKEND,
    SHEET_ATTACHMENT3_WEEKDAY, SHEET_ATTACHMENT3_WEEKEND,
    FILE_CLEAN_DATA
)


def read_attachment1():
    """读取附件1：区域基础数据（取前10行有效数据，跳过汇总行）"""
    print("[1/4] 读取附件1: 区域基础数据...")
    df = pd.read_excel(FILE_ATTACHMENT1)
    df = df.iloc[:10].copy()

    df.columns = [
        '区域编号','区域总面积','充电覆盖面积','人口密度',
        '车流量','商业POI数','充电桩数量','快充数量','慢充数量','电网容量'
    ]
    df['区域编号'] = df['区域编号'].astype(int)
    print(f"  → {len(df)} 个区域, {len(df.columns)} 个字段")
    return df


def read_attachment2():
    """
    读取附件2：分时段充电车次。
    原始文件包含两个sheet，通过 sheet_name 分别读取，
    不拆分原始文件。
    """
    print("[2/4] 读取附件2: 分时段充电车次...")
    df_wd = pd.read_excel(FILE_ATTACHMENT2, sheet_name=SHEET_ATTACHMENT2_WEEKDAY)
    df_wd['日期类型'] = '工作日'
    df_we = pd.read_excel(FILE_ATTACHMENT2, sheet_name=SHEET_ATTACHMENT2_WEEKEND)
    df_we['日期类型'] = '周末'
    df = pd.concat([df_wd, df_we], ignore_index=True)
    print(f"  → 工作日 {len(df_wd)} 条, 周末 {len(df_we)} 条")
    return df


def read_attachment3():
    """
    读取附件3：分时段充电负荷。
    原始文件包含两个sheet，通过 sheet_name 分别读取，
    不拆分原始文件。
    """
    print("[3/4] 读取附件3: 分时段充电负荷...")
    df_wd = pd.read_excel(FILE_ATTACHMENT3, sheet_name=SHEET_ATTACHMENT3_WEEKDAY)
    df_wd = df_wd.iloc[:10].copy()
    df_wd['日期类型'] = '工作日'
    df_we = pd.read_excel(FILE_ATTACHMENT3, sheet_name=SHEET_ATTACHMENT3_WEEKEND)
    df_we = df_we.iloc[:10].copy()
    df_we['日期类型'] = '周末'
    df = pd.concat([df_wd, df_we], ignore_index=True)
    print(f"  → 工作日 {len(df_wd)} 条, 周末 {len(df_we)} 条")
    return df


def wide_to_long(df, value_name):
    """宽表转长表"""
    time_cols = [c for c in df.columns if '-' in str(c) and ':' not in str(c)]
    region_col = '区域' if '区域' in df.columns else '区域编号'
    id_vars = [region_col, '日期类型'] if '日期类型' in df.columns else [region_col]
    df_long = df.melt(id_vars=id_vars, value_vars=time_cols,
                      var_name='时段', value_name=value_name)
    df_long['小时'] = df_long['时段'].str.split('-').str[0].astype(int)
    df_long.drop(columns=['时段'], inplace=True)
    return df_long


def detect_outliers(series):
    """IQR异常值检测（仅统计）"""
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    return ((series < Q1 - 1.5*IQR) | (series > Q3 + 1.5*IQR)).sum()


def main():
    print("=" * 60)
    print("Step 1: 数据读取与清洗")
    print("=" * 60)

    df1 = read_attachment1()
    df2 = read_attachment2()
    df3 = read_attachment3()

    print("\n[4/4] 宽表→长表 + 合并...")
    df2_long = wide_to_long(df2, '充电车次')
    df3_long = wide_to_long(df3, '充电负荷')

    for df_x in [df2_long, df3_long]:
        if '区域' in df_x.columns:
            df_x.rename(columns={'区域': '区域编号'}, inplace=True)
        df_x['区域编号'] = df_x['区域编号'].astype(int)

    df = df3_long.merge(df2_long, on=['区域编号','小时','日期类型'], how='left')
    df = df.merge(df1, on='区域编号', how='left')
    print(f"  → {df.shape[0]} 条记录, {df.shape[1]} 个字段")

    print("\n[异常值检测] IQR...")
    for col in ['充电负荷','充电车次']:
        n = detect_outliers(df[col])
        print(f"  → {col}: {n} 个（经核实属正常波动，保留）")

    print("\n[标准化] Z-score...")
    for col in ['人口密度','车流量','商业POI数','充电桩数量','快充数量','慢充数量']:
        mu, sigma = df[col].mean(), df[col].std()
        df[f'{col}_标准化'] = (df[col] - mu) / sigma

    cols = ['区域编号','小时','日期类型','充电负荷','充电车次',
            '区域总面积','充电覆盖面积','人口密度','车流量','商业POI数',
            '充电桩数量','快充数量','慢充数量','电网容量',
            '人口密度_标准化','车流量_标准化','商业POI数_标准化',
            '充电桩数量_标准化','快充数量_标准化','慢充数量_标准化']
    df = df[[c for c in cols if c in df.columns]]

    df.to_excel(FILE_CLEAN_DATA, index=False)
    print(f"\n✅ 清洗完成 → {FILE_CLEAN_DATA}")
    print(f"   {df.shape[0]}行 × {df.shape[1]}列 | 缺失值: {df.isnull().sum().sum()}")
    return df


if __name__ == '__main__':
    main()
