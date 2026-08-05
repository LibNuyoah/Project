"""
Step 7: 结果汇总与未来需求预测
-----------------------------
汇总前序所有分析结果，生成最终预测结果。
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
    RESULTS_DIR, RESULTS_FIGURES, RESULTS_TABLES, MODELS_DIR,
    FILE_CLEAN_DATA, FILE_PREDICTION_RESULT, FILE_HOURLY_PREDICTION,
    FILE_MODEL_COMPARISON, FILE_XGBOOST_METRICS, FILE_XGBOOST_MODEL
)
from src.problem1.analysis.region_type_loader import get_region_types, get_region_names
REGION_NAMES = get_region_names()
REGION_TYPES = get_region_types()


def load_model():
    """加载训练好的XGBoost模型"""
    with open(FILE_XGBOOST_MODEL, 'rb') as f:
        model = pickle.load(f)
    return model


def predict_daily_demand(model):
    """
    预测各区域未来短期日均充电需求。
    """
    print("=" * 60)
    print("未来充电需求预测")
    print("=" * 60)

    df = pd.read_excel(FILE_CLEAN_DATA)

    region_info = df.groupby('区域编号').agg({
        '区域总面积': 'first', '充电覆盖面积': 'first',
        '人口密度': 'first', '车流量': 'first',
        '商业POI数': 'first', '充电桩数量': 'first',
        '快充数量': 'first', '慢充数量': 'first',
        '电网容量': 'first'
    }).reset_index()

    results = []

    for _, row in region_info.iterrows():
        region_id = int(row['区域编号'])
        for day_type in ['工作日', '周末']:
            for hour in range(24):
                features = build_feature_vector(row, hour, day_type)
                pred_load = model.predict(features)[0]
                results.append({
                    '区域编号': region_id,
                    '小时': hour,
                    '日期类型': day_type,
                    '预测负荷': max(0, pred_load)
                })

    pred_df = pd.DataFrame(results)

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

    print("\n[各区域未来短期日均充电需求预测]")
    print(daily_summary[['区域编号', '区域名称', '区域类型',
                         '预测日均需求_kWh', '峰值负荷_kWh', '谷值负荷_kWh']].to_string(index=False))

    return daily_summary, pred_df


def build_feature_vector(region_row, hour, day_type):
    """构建与训练时一致的特征向量（已剔除充电覆盖面积，共33维）"""
    continuous = [
        region_row['人口密度'], region_row['车流量'], region_row['商业POI数'],
        region_row['充电桩数量'], region_row['快充数量'], region_row['慢充数量'],
        region_row['电网容量'], region_row['区域总面积']
    ]
    is_weekday = 1 if day_type == '工作日' else 0
    hour_oh = [0] * 24
    hour_oh[hour] = 1
    features = np.array([continuous + [is_weekday] + hour_oh])
    return features


def create_model_comparison():
    """创建模型评价汇总表"""
    print("\n" + "=" * 60)
    print("XGBoost 模型评价汇总")
    print("=" * 60)

    xgb_metrics = pd.read_excel(FILE_XGBOOST_METRICS)

    comparison = xgb_metrics[xgb_metrics['数据集'].isin(['训练集', '测试集'])][
        ['数据集', 'MAE(kWh)', 'RMSE(kWh)', 'R2', 'MAPE(%)']
    ].copy()

    print("\n" + comparison.to_string(index=False))

    comparison.to_excel(FILE_MODEL_COMPARISON, index=False)
    print(f"\n模型评价表已保存: {FILE_MODEL_COMPARISON}")

    return comparison


def plot_final_summary(daily_summary, comparison, pred_df):
    """绘制最终汇总图"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ── 图1: 各区域预测日均需求 ──
    ax1 = axes[0, 0]
    names = daily_summary['区域名称'].tolist()
    demands = daily_summary['预测日均需求_kWh'].values
    types_list = daily_summary['区域类型'].tolist()

    type_colors_map = {
        '老城核心区': '#E74C3C', '城市新区': '#3498DB',
        '工业区': '#9B59B6', '文旅区': '#2ECC71', '城郊过渡区': '#95A5A6',
        '城郊/工业区': '#95A5A6'
    }
    bar_colors = [type_colors_map[t] for t in types_list]

    bars = ax1.bar(range(len(names)), demands, color=bar_colors, edgecolor='white')
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('预测日均充电需求 (kWh)', fontsize=11)
    ax1.set_title('各区域未来短期日均充电需求预测', fontsize=13, fontweight='bold')

    for bar, val in zip(bars, demands):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                 f'{val:,.0f}', ha='center', fontsize=8)

    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=c, label=t) for t, c in type_colors_map.items()]
    ax1.legend(handles=legend_patches, loc='upper right', fontsize=8)

    # ── 图2: XGBoost 评价指标 ──
    ax2 = axes[0, 1]
    metrics_names = ['MAE\n(kWh)', 'RMSE\n(kWh)', 'R2']
    train_vals = [
        comparison[comparison['数据集'] == '训练集']['MAE(kWh)'].values[0],
        comparison[comparison['数据集'] == '训练集']['RMSE(kWh)'].values[0],
        comparison[comparison['数据集'] == '训练集']['R2'].values[0],
    ]
    test_vals = [
        comparison[comparison['数据集'] == '测试集']['MAE(kWh)'].values[0],
        comparison[comparison['数据集'] == '测试集']['RMSE(kWh)'].values[0],
        comparison[comparison['数据集'] == '测试集']['R2'].values[0],
    ]

    x = np.arange(len(metrics_names))
    width = 0.3
    ax2.bar(x - width/2, train_vals, width, label='训练集', color='#E74C3C', edgecolor='white')
    ax2.bar(x + width/2, test_vals, width, label='测试集', color='#3498DB', edgecolor='white')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics_names, fontsize=11)
    ax2.set_title('XGBoost 训练集 vs 测试集指标对比', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)

    for i, (tv, tev) in enumerate(zip(train_vals, test_vals)):
        ax2.text(i - width/2, tv + max(train_vals)*0.02, f'{tv:.2f}', ha='center', fontsize=8)
        ax2.text(i + width/2, tev + max(test_vals)*0.02, f'{tev:.2f}', ha='center', fontsize=8)

    # ── 图3: 工作日 vs 周末24小时预测负荷 ──
    ax3 = axes[1, 0]
    wd_hourly = pred_df[pred_df['日期类型'] == '工作日'].groupby('小时')['预测负荷'].sum()
    we_hourly = pred_df[pred_df['日期类型'] == '周末'].groupby('小时')['预测负荷'].sum()

    ax3.plot(wd_hourly.index, wd_hourly.values, 'o-', color='#E74C3C',
             linewidth=2, markersize=5, label='工作日')
    ax3.plot(we_hourly.index, we_hourly.values, 's--', color='#3498DB',
             linewidth=2, markersize=5, label='周末')
    ax3.set_xlabel('小时', fontsize=11)
    ax3.set_ylabel('预测总充电负荷 (kWh)', fontsize=11)
    ax3.set_title('预测 24小时总充电负荷曲线 (工作日 vs 周末)', fontsize=13, fontweight='bold')
    ax3.set_xticks(range(0, 24, 3))
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, linestyle='--')

    # ── 图4: 各区域类型需求占比 ──
    ax4 = axes[1, 1]
    type_demand = daily_summary.groupby('区域类型')['预测日均需求_kWh'].sum().sort_values(ascending=False)
    explode = [0.05 if i == 0 else 0 for i in range(len(type_demand))]
    wedges, texts, autotexts = ax4.pie(
        type_demand.values, labels=type_demand.index, autopct='%1.1f%%',
        colors=[type_colors_map[t] for t in type_demand.index],
        explode=explode, startangle=90, textprops={'fontsize': 9}
    )
    ax4.set_title('各区域类型充电需求占比', fontsize=13, fontweight='bold')

    plt.tight_layout()
    output_path = os.path.join(RESULTS_FIGURES, 'prediction_summary.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n最终预测汇总图已保存: {output_path}")


def main():
    # 加载模型
    model = load_model()

    # 预测
    daily_summary, pred_df = predict_daily_demand(model)

    # 保存预测结果
    output_cols = ['区域编号', '区域名称', '区域类型',
                   '预测日均需求_kWh', '预测日均需求_MWh',
                   '工作日日均需求_kWh', '周末日均需求_kWh',
                   '峰值负荷_kWh', '谷值负荷_kWh']
    daily_output = daily_summary[output_cols].copy()
    daily_output.to_excel(FILE_PREDICTION_RESULT, index=False)
    print(f"\n预测结果已保存: {FILE_PREDICTION_RESULT}")

    # 保存详细分时段预测
    pred_df.to_excel(FILE_HOURLY_PREDICTION, index=False)
    print(f"分时段预测已保存: {FILE_HOURLY_PREDICTION}")

    # 模型对比
    comparison = create_model_comparison()

    # 绘图
    plot_final_summary(daily_summary, comparison, pred_df)

    # ── 问题1关键变量输出 ──
    print("\n" + "=" * 60)
    print("问题1关键指标")
    print("=" * 60)

    top_region = daily_summary.iloc[0]
    bottom_region = daily_summary.iloc[-1]
    total_daily_energy = daily_summary['预测日均需求_kWh'].sum()
    region_num = len(daily_summary)
    region_daily_demand = daily_summary.set_index('区域名称')['预测日均需求_kWh'].to_dict()
    max_region = top_region['区域名称']
    max_value = top_region['预测日均需求_kWh']
    min_region = bottom_region['区域名称']
    region_ratio = top_region['预测日均需求_kWh'] / bottom_region['预测日均需求_kWh']

    wd_hourly = pred_df[pred_df['日期类型'] == '工作日'].groupby('小时')['预测负荷'].sum()
    we_hourly = pred_df[pred_df['日期类型'] == '周末'].groupby('小时')['预测负荷'].sum()
    peak_hours = wd_hourly.idxmax()
    valley_hours = wd_hourly.idxmin()
    workday_mean = daily_summary['工作日日均需求_kWh'].mean()
    weekend_mean = daily_summary['周末日均需求_kWh'].mean()

    print("\n区域数量:", region_num)
    print("\n全市日均充电需求:", total_daily_energy)
    print("\n各区域日均需求:")
    print(region_daily_demand)
    print("\n最大需求区域:", max_region)
    print("\n最大需求值:", max_value)
    print("\n最小需求区域:", min_region)
    print("\n区域需求比:", region_ratio)
    print("\n峰值时间:", peak_hours)
    print("\n谷值时间:", valley_hours)
    print("\n工作日平均需求:", workday_mean)
    print("\n周末平均需求:", weekend_mean)

    return daily_summary, comparison


if __name__ == '__main__':
    daily_summary, comparison = main()
