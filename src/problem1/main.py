"""
Step 7: 结果汇总与充电需求估计
-----------------------------
基于训练好的XGBoost模型（GroupKFold区域泛化验证），
预测各区域典型工作日/周末的充电需求。
输出: results/prediction_result.xlsx
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import pickle
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from utils.paths import (
    RESULTS_FIGURES, FILE_CLEAN_DATA, FILE_PREDICTION_RESULT,
    FILE_HOURLY_PREDICTION, FILE_XGBOOST_METRICS, FILE_XGBOOST_MODEL
)
from src.problem1.analysis.region_type_loader import get_region_types, get_region_names
# 直接复用xgboost_model的特征工程，确保与训练时完全一致
from src.problem1.model.xgboost_model import prepare_features, load_data
REGION_NAMES = get_region_names()
REGION_TYPES = get_region_types()


def load_model():
    """加载训练好的XGBoost模型"""
    with open(FILE_XGBOOST_MODEL, 'rb') as f:
        model = pickle.load(f)
    return model


def predict_daily_demand(model):
    """
    使用已训练模型预测各区域充电需求。
    直接调用 train 时的 prepare_features 生成一致的特征矩阵，
    再用模型进行批量预测。
    """
    print("=" * 60)
    print("区域充电需求估计（GroupKFold泛化模型）")
    print("=" * 60)

    # 加载数据并使用与训练完全一致的特征工程
    df = load_data()
    X, y, feature_names, groups, _ = prepare_features(df)

    # 批量预测（log1p训练，expm1恢复到kWh）
    y_pred_log = model.predict(X)
    y_pred_all = np.expm1(y_pred_log)
    y_pred_all = np.maximum(y_pred_all, 0)

    # 构建预测结果DataFrame
    df = df.sort_values(['区域编号', '日期类型', '小时']).reset_index(drop=True)
    pred_df = df[['区域编号', '小时', '日期类型']].copy()
    pred_df['预测负荷'] = y_pred_all

    # 汇总为日均需求
    daily_summary = pred_df.groupby('区域编号').agg(
        预测日均需求_kWh=('预测负荷', 'sum'),
        工作日日均需求_kWh=('预测负荷', lambda x: x[pred_df.loc[x.index, '日期类型'] == '工作日'].sum()),
        周末日均需求_kWh=('预测负荷', lambda x: x[pred_df.loc[x.index, '日期类型'] == '周末'].sum()),
        峰值负荷_kWh=('预测负荷', 'max'),
        谷值负荷_kWh=('预测负荷', 'min'),
    ).reset_index()

    daily_summary['预测日均需求_kWh'] = (
        daily_summary['工作日日均需求_kWh'] + daily_summary['周末日均需求_kWh']
    ) / 2

    daily_summary['区域名称'] = daily_summary['区域编号'].map(REGION_NAMES)
    daily_summary['区域类型'] = daily_summary['区域编号'].map(REGION_TYPES)
    daily_summary['预测日均需求_MWh'] = daily_summary['预测日均需求_kWh'] / 1000

    daily_summary = daily_summary.sort_values('预测日均需求_kWh', ascending=False)

    print("\n[各区域充电需求估计结果]")
    print(daily_summary[['区域编号', '区域名称', '区域类型',
                         '预测日均需求_kWh', '峰值负荷_kWh', '谷值负荷_kWh']].to_string(index=False))

    return daily_summary, pred_df


def plot_final_summary(daily_summary, pred_df):
    """绘制最终汇总图"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    type_colors_map = {
        '老城核心区': '#E74C3C', '城市新区': '#3498DB',
        '工业区': '#9B59B6', '文旅区': '#2ECC71', '城郊过渡区': '#95A5A6',
        '城郊/工业区': '#95A5A6'
    }

    # 图1: 各区域预测日均需求
    ax1 = axes[0, 0]
    names = daily_summary['区域名称'].tolist()
    demands = daily_summary['预测日均需求_kWh'].values
    types_list = daily_summary['区域类型'].tolist()
    bar_colors = [type_colors_map[t] for t in types_list]
    bars = ax1.bar(range(len(names)), demands, color=bar_colors, edgecolor='white')
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('预测日均充电需求 (kWh)', fontsize=11)
    ax1.set_title('各区域日均充电需求估计', fontsize=13, fontweight='bold')
    for bar, val in zip(bars, demands):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                 f'{val:,.0f}', ha='center', fontsize=8)
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=c, label=t) for t, c in type_colors_map.items()]
    ax1.legend(handles=legend_patches, loc='upper right', fontsize=8)

    # 图2: 工作日 vs 周末24小时负荷曲线
    ax2 = axes[0, 1]
    wd_hourly = pred_df[pred_df['日期类型'] == '工作日'].groupby('小时')['预测负荷'].sum()
    we_hourly = pred_df[pred_df['日期类型'] == '周末'].groupby('小时')['预测负荷'].sum()
    ax2.plot(wd_hourly.index, wd_hourly.values, 'o-', color='#E74C3C',
             linewidth=2, markersize=5, label='工作日')
    ax2.plot(we_hourly.index, we_hourly.values, 's--', color='#3498DB',
             linewidth=2, markersize=5, label='周末')
    ax2.set_xlabel('小时', fontsize=11)
    ax2.set_ylabel('总充电负荷 (kWh)', fontsize=11)
    ax2.set_title('24小时充电负荷曲线', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(0, 24, 3)); ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')

    # 图3: 各区域类型需求占比
    ax3 = axes[1, 0]
    type_demand = daily_summary.groupby('区域类型')['预测日均需求_kWh'].sum().sort_values(ascending=False)
    explode = [0.05 if i == 0 else 0 for i in range(len(type_demand))]
    ax3.pie(type_demand.values, labels=type_demand.index, autopct='%1.1f%%',
            colors=[type_colors_map[t] for t in type_demand.index],
            explode=explode, startangle=90, textprops={'fontsize': 9})
    ax3.set_title('各区域类型充电需求占比', fontsize=13, fontweight='bold')

    # 图4: 模型信息
    ax4 = axes[1, 1]
    ax4.axis('off')
    ax4.text(0.1, 0.5, '基于区域泛化能力的\n充电需求估计模型\n\n验证：GroupKFold(n_splits=10)\n特征：40维（空间+时间+聚类+互补）',
             fontsize=12, verticalalignment='center')

    plt.tight_layout()
    output_path = os.path.join(RESULTS_FIGURES, 'prediction_summary.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n预测汇总图已保存: {output_path}")


def main():
    model = load_model()
    daily_summary, pred_df = predict_daily_demand(model)

    # 保存预测结果
    output_cols = ['区域编号', '区域名称', '区域类型',
                   '预测日均需求_kWh', '预测日均需求_MWh',
                   '工作日日均需求_kWh', '周末日均需求_kWh',
                   '峰值负荷_kWh', '谷值负荷_kWh']
    daily_output = daily_summary[output_cols].copy()
    daily_output.to_excel(FILE_PREDICTION_RESULT, index=False)
    print(f"\n预测结果已保存: {FILE_PREDICTION_RESULT}")

    pred_df.to_excel(FILE_HOURLY_PREDICTION, index=False)
    print(f"分时段预测已保存: {FILE_HOURLY_PREDICTION}")

    # 绘图
    plot_final_summary(daily_summary, pred_df)

    # ── 问题1关键变量输出 ──
    print("\n" + "=" * 60)
    print("问题1关键指标")
    print("=" * 60)
    top_region = daily_summary.iloc[0]
    bottom_region = daily_summary.iloc[-1]
    total_daily_energy = daily_summary['预测日均需求_kWh'].sum()
    region_daily_demand = daily_summary.set_index('区域名称')['预测日均需求_kWh'].to_dict()
    wd_hourly = pred_df[pred_df['日期类型'] == '工作日'].groupby('小时')['预测负荷'].sum()

    print("\n区域数量:", len(daily_summary))
    print("\n全市日均充电需求:", total_daily_energy)
    print("\n各区域日均需求:")
    print(region_daily_demand)
    print("\n最大需求区域:", top_region['区域名称'])
    print("\n最大需求值:", top_region['预测日均需求_kWh'])
    print("\n最小需求区域:", bottom_region['区域名称'])
    print("\n区域需求比:", top_region['预测日均需求_kWh'] / bottom_region['预测日均需求_kWh'])
    print("\n峰值时间:", wd_hourly.idxmax())
    print("\n谷值时间:", wd_hourly.idxmin())
    print("\n工作日平均需求:", daily_summary['工作日日均需求_kWh'].mean())
    print("\n周末平均需求:", daily_summary['周末日均需求_kWh'].mean())

    return daily_summary


if __name__ == '__main__':
    main()
