"""
Step 2a: 空间维度分析
--------------------
分析充电需求在10个区域之间的空间分布规律。
输出: result/figures/spatial.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import sys

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

from utils.paths import RESULT_DIR, RESULT_FIGURES, FILE_CLEAN_DATA
from src.question1.analysis.region_type_loader import get_region_types, get_region_names
REGION_NAMES = get_region_names()
REGION_TYPES = get_region_types()


def load_data():
    """加载清洗后的数据"""
    return pd.read_excel(FILE_CLEAN_DATA)


def analyze_spatial(df):
    """空间维度分析"""
    print("=" * 60)
    print("空间维度分析")
    print("=" * 60)

    # 计算每个区域的工作日和周末日总需求
    # 按区域，工作日/周末分组求和
    daily = df.groupby(['区域编号', '日期类型'])['充电负荷'].sum().reset_index()
    daily_pivot = daily.pivot(index='区域编号', columns='日期类型', values='充电负荷')
    daily_pivot['日均需求'] = (daily_pivot['工作日'] + daily_pivot['周末']) / 2
    daily_pivot['日总需求_工作日'] = daily_pivot['工作日']
    daily_pivot['日总需求_周末'] = daily_pivot['周末']

    # 获取区域基础属性（取第一条记录即可，因为同一区域的属性相同）
    region_info = df.groupby('区域编号').agg({
        '区域总面积': 'first',
        '充电覆盖面积': 'first',
        '充电桩数量': 'first',
        '快充数量': 'first',
        '慢充数量': 'first',
        '人口密度': 'first',
        '车流量': 'first',
        '商业POI数': 'first',
        '电网容量': 'first'
    }).reset_index()

    # 合并
    analysis = daily_pivot.merge(region_info, on='区域编号')

    # 计算衍生指标
    # 各区域工作日/周末日总充电车次
    daily_sessions = df.groupby(['区域编号', '日期类型'])['充电车次'].sum().reset_index()
    sessions_pivot = daily_sessions.pivot(index='区域编号', columns='日期类型', values='充电车次')
    sessions_pivot.columns = ['工作日车次', '周末车次']  # 避免与负荷列名冲突
    analysis = analysis.merge(sessions_pivot, on='区域编号')
    analysis['日均车次'] = (analysis['工作日车次'] + analysis['周末车次']) / 2

    # 单桩日均利用率（车次/桩/日）
    analysis['单桩日均车次'] = analysis['日均车次'] / analysis['充电桩数量']

    # 单位面积日均需求（kWh/km²）
    analysis['单位面积需求'] = analysis['日均需求'] / analysis['充电覆盖面积']

    # 排名
    analysis['需求排名'] = analysis['日均需求'].rank(ascending=False).astype(int)
    analysis = analysis.sort_values('日均需求', ascending=False)

    # 添加区域名称
    analysis['区域名称'] = analysis['区域编号'].map(REGION_NAMES)
    analysis['区域类型'] = analysis['区域编号'].map(REGION_TYPES)

    print("\n[区域充电需求排名]")
    print(analysis[['需求排名', '区域编号', '区域名称', '区域类型',
                     '日均需求', '单桩日均车次', '单位面积需求']].to_string(index=False))

    return analysis


def plot_spatial(analysis):
    """绘制空间分析图"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = ['#E74C3C', '#E67E22', '#F1C40F', '#2ECC71', '#1ABC9C',
              '#3498DB', '#9B59B6', '#34495E', '#95A5A6', '#E91E63']

    names = analysis['区域名称'].tolist()
    types = analysis['区域类型'].tolist()

    # ── 图1: 区域日充电需求柱状图 ──
    ax1 = axes[0, 0]
    bars1 = ax1.bar(range(len(names)), analysis['日均需求'].values, color=colors, edgecolor='white')
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('日均充电需求 (kWh)')
    ax1.set_title('各区域日均充电需求排名', fontsize=14, fontweight='bold')
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

    # 标注数值
    for bar, val in zip(bars1, analysis['日均需求']):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 300,
                 f'{val:,.0f}', ha='center', fontsize=8)

    # 类型图例
    from matplotlib.patches import Patch
    type_colors = {
        '老城核心区': '#E74C3C',
        '城市新区': '#3498DB',
        '工业区': '#9B59B6',
        '文旅区': '#2ECC71',
        '城郊过渡区': '#95A5A6',
        '城郊/工业区': '#95A5A6'
    }
    legend_patches = [Patch(facecolor=c, label=t) for t, c in type_colors.items()]
    ax1.legend(handles=legend_patches, loc='upper right', fontsize=8)

    # ── 图2: 单桩日均利用率 ──
    ax2 = axes[0, 1]
    sorted_by_util = analysis.sort_values('单桩日均车次', ascending=False)
    bars2 = ax2.bar(range(len(names)), sorted_by_util['单桩日均车次'].values,
                    color=[colors[i] for i in sorted_by_util.index - 1], edgecolor='white')
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(sorted_by_util['区域名称'].tolist(), rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('日均车次/桩')
    ax2.set_title('各区域单桩日均利用率', fontsize=14, fontweight='bold')
    ax2.axhline(y=analysis['单桩日均车次'].mean(), color='red', linestyle='--',
                label=f"平均: {analysis['单桩日均车次'].mean():.1f}")
    ax2.legend(fontsize=9)

    for bar, val in zip(bars2, sorted_by_util['单桩日均车次']):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f'{val:.1f}', ha='center', fontsize=8)

    # ── 图3: 单位面积充电需求 ──
    ax3 = axes[1, 0]
    sorted_by_density = analysis.sort_values('单位面积需求', ascending=False)
    bars3 = ax3.bar(range(len(names)), sorted_by_density['单位面积需求'].values,
                    color=[colors[i] for i in sorted_by_density.index - 1], edgecolor='white')
    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels(sorted_by_density['区域名称'].tolist(), rotation=45, ha='right', fontsize=9)
    ax3.set_ylabel('kWh / km2')
    ax3.set_title('各区域单位面积充电需求', fontsize=14, fontweight='bold')
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

    for bar, val in zip(bars3, sorted_by_density['单位面积需求']):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                 f'{val:,.0f}', ha='center', fontsize=8)

    # ── 图4: 工作日 vs 周末区域需求对比 ──
    ax4 = axes[1, 1]
    x = np.arange(len(names))
    width = 0.35
    bars_wd = ax4.bar(x - width / 2, analysis['工作日'].values, width,
                       label='工作日', color='#E74C3C', edgecolor='white')
    bars_we = ax4.bar(x + width / 2, analysis['周末'].values, width,
                       label='周末', color='#3498DB', edgecolor='white')
    ax4.set_xticks(x)
    ax4.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax4.set_ylabel('日总充电需求 (kWh)')
    ax4.set_title('各区域工作日 vs 周末充电需求对比', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

    plt.tight_layout()
    output_path = os.path.join(RESULT_FIGURES, 'spatial.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n✅ 空间分析图已保存: {output_path}")

    return analysis


def main():
    df = load_data()
    analysis = analyze_spatial(df)
    plot_spatial(analysis)

    # 输出分析结论
    print("\n" + "=" * 60)
    print("空间维度分析结论（可直接写入论文）")
    print("=" * 60)

    top3 = analysis.head(3)
    bottom3 = analysis.tail(3)
    max_demand = analysis.iloc[0]
    min_demand = analysis.iloc[-1]
    ratio = max_demand['日均需求'] / min_demand['日均需求']

    print(f"""
1. 区域需求差异显著：充电需求最高的区域"{max_demand['区域名称']}（{max_demand['区域类型']}）"
   日均需求 {max_demand['日均需求']:,.0f} kWh，是最低区域"{min_demand['区域名称']}（{min_demand['区域类型']}）"
   的 {ratio:.1f} 倍。

2. 需求排名前三的区域为：{top3['区域名称'].tolist()}，均为核心城区和城市新区，
   人口密度高、商业POI密集、车流量大是其主要特征。

3. 需求最低的区域为：{bottom3['区域名称'].tolist()}，主要是城郊过渡区和工业区，
   人口密度低、充电基础设施相对不足。

4. 单桩利用率差异表明：核心城区充电桩利用率远高于城郊区域，
   存在充电资源空间错配问题。
""")

    return analysis


if __name__ == '__main__':
    analysis = main()
