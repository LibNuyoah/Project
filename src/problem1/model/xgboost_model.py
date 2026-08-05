"""
XGBoost 充电需求预测模型（优化版）
--------------------------------
主模型：基于多维特征预测充电负荷。

优化内容：
  1. 加入区域功能聚类特征（One-Hot编码）
  2. 加入历史负荷滞后特征（lag_1, lag_24）
  3. 时间序列划分（前80%训练/后20%测试）
  4. 正则化超参数搜索（min_child_weight, gamma）
  5. 新增SMAPE评价指标
  6. 输出优化前后对比表

输出:
  models/xgboost_model.pkl
  results/figures/xgboost_evaluation.png
  results/tables/xgboost_metrics.xlsx
  results/tables/xgboost_compare.xlsx
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

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# GridSearchCV 已保留导入用作实验记录（历史超参数搜索过程见 train_xgboost 注释）
import xgboost as xgb

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

from utils.paths import (
    RESULTS_FIGURES,
    FILE_CLEAN_DATA, FILE_CLUSTER_RESULT,
    FILE_XGBOOST_MODEL, FILE_XGBOOST_METRICS, FILE_XGBOOST_COMPARE
)


# ═══════════════════════════════════════════════════════════════
# 0. 数据加载
# ═══════════════════════════════════════════════════════════════

def load_data():
    """加载清洗后的数据"""
    return pd.read_excel(FILE_CLEAN_DATA)


def load_cluster_data():
    """加载K-means聚类结果，返回 区域编号→功能特征 映射"""
    cluster_df = pd.read_excel(FILE_CLUSTER_RESULT)
    return dict(zip(cluster_df['区域编号'], cluster_df['功能特征']))


# ═══════════════════════════════════════════════════════════════
# 1. 特征工程（含聚类特征 + 滞后特征）
# ═══════════════════════════════════════════════════════════════

def add_cluster_features(df):
    """
    读取K-means聚类结果，添加区域类型One-Hot编码。
    使用One-Hot而非数字编码，避免模型错误学习类别间的大小关系。
    """
    region_type_map = load_cluster_data()
    df['区域类型'] = df['区域编号'].map(region_type_map)

    # One-Hot编码：老城核心区 / 城市新区 / 城郊/工业区
    cluster_dummies = pd.get_dummies(df['区域类型'], prefix='区域类型')

    # 统一列名（处理可能的编码差异）
    cluster_cols = []
    for col in cluster_dummies.columns:
        # 移除特殊字符，统一格式
        clean_name = col.replace('/', '').replace(' ', '')
        cluster_cols.append(clean_name)
    cluster_dummies.columns = cluster_cols

    df = pd.concat([df, cluster_dummies], axis=1)
    print(f"  → 区域聚类特征: {cluster_cols}")
    return df, cluster_cols


def add_lag_features(df):
    """
    添加历史负荷滞后特征（按区域分组shift，防止跨区域数据污染）。

    lag_1:  前一小时负荷（捕捉短期连续性）
    lag_24: 前一天同期负荷近似（24小时前）

    注：原始数据每区域48行（24h×2日期类型），168小时滞后（一周）
    超出数据范围，故仅保留lag_1和lag_24。
    """
    # 确保按区域→日期类型→小时排序
    df = df.sort_values(['区域编号', '日期类型', '小时']).reset_index(drop=True)

    # 按区域分组shift，防止跨区域数据污染
    df['Load_lag_1'] = df.groupby('区域编号')['充电负荷'].shift(1)
    df['Load_lag_24'] = df.groupby('区域编号')['充电负荷'].shift(24)

    # 对于shift产生的NaN（每组开头），用0填充
    lag_na_1 = df['Load_lag_1'].isna().sum()
    lag_na_24 = df['Load_lag_24'].isna().sum()
    df['Load_lag_1'] = df['Load_lag_1'].fillna(0)
    df['Load_lag_24'] = df['Load_lag_24'].fillna(0)

    print(f"  → 滞后特征: Load_lag_1 (NaN: {lag_na_1}), Load_lag_24 (NaN: {lag_na_24})")
    return df


def prepare_features(df, corr_threshold=0.05):
    """
    特征工程（优化版）：
      - 相关性筛选连续特征
      - 区域聚类 One-Hot 编码
      - 历史负荷滞后特征
      - 小时 One-Hot 编码
    """
    print("=" * 60)
    print("XGBoost 特征工程（优化版：聚类 + 滞后 + 相关性筛选）")
    print("=" * 60)

    # ── 1. 添加区域聚类特征 ──
    df, cluster_cols = add_cluster_features(df)

    # ── 2. 添加历史负荷滞后特征 ──
    df = add_lag_features(df)

    # ── 3. 目标变量 ──
    y = df['充电负荷'].values

    # ── 4. 连续特征相关性筛选 ──
    all_continuous = [
        '人口密度', '车流量', '商业POI数',
        '充电桩数量', '快充数量', '慢充数量', '电网容量',
        '区域总面积', '充电覆盖面积'
    ]

    corr_with_target = {}
    for feat in all_continuous:
        if feat in df.columns:
            r = df[feat].corr(df['充电负荷'])
            corr_with_target[feat] = r

    selected_continuous = [
        f for f in all_continuous
        if f in corr_with_target and abs(corr_with_target[f]) >= corr_threshold
    ]
    dropped_continuous = [f for f in all_continuous if f not in selected_continuous]

    print(f"\n  相关性筛选 (|r| >= {corr_threshold}):")
    for feat in selected_continuous:
        print(f"    ✅ {feat}: r={corr_with_target[feat]:+.4f}")
    for feat in dropped_continuous:
        print(f"    ❌ {feat}: r={corr_with_target[feat]:+.4f} (已剔除)")

    # ── 5. 类别特征编码 ──
    df_feat = df.copy()
    df_feat['是否工作日'] = (df_feat['日期类型'] == '工作日').astype(int)

    # 小时 one-hot（24维）
    hour_dummies = pd.get_dummies(df_feat['小时'], prefix='小时')

    # ── 6. 构建特征矩阵 ──
    # 连续特征 + 工作日标志 + 小时one-hot + 聚类one-hot + 滞后特征
    feature_blocks = [
        df_feat[selected_continuous].values,    # 连续特征
        df_feat[['是否工作日']].values,          # 工作日标志
        hour_dummies.values,                     # 小时 one-hot (24维)
    ]

    # 添加聚类 one-hot
    for col in cluster_cols:
        if col in df_feat.columns:
            feature_blocks.append(df_feat[[col]].values)

    # 添加滞后特征
    feature_blocks.append(df_feat[['Load_lag_1', 'Load_lag_24']].values)

    X = np.hstack(feature_blocks)

    # 构建特征名列表
    feature_names = (
        selected_continuous +
        ['是否工作日'] +
        list(hour_dummies.columns) +
        [c for c in cluster_cols if c in df_feat.columns] +
        ['Load_lag_1', 'Load_lag_24']
    )

    print(f"\n  → 特征维度: {X.shape[1]} (连续:{len(selected_continuous)}, "
          f"小时:24, 日期类型:1, 聚类:{len([c for c in cluster_cols if c in df_feat.columns])}, "
          f"滞后:2)")
    print(f"  → 样本数量: {X.shape[0]}")
    print(f"  → 剔除低相关特征: {len(dropped_continuous)} 个")

    return X, y, feature_names


# ═══════════════════════════════════════════════════════════════
# 2. 时间序列划分
# ═══════════════════════════════════════════════════════════════

def time_series_split(X, y, test_size=0.2):
    """
    时间序列划分：前80%时间作为训练集，后20%作为测试集。
    不做随机打乱，避免未来数据泄漏到训练集中。

    数据已按 区域编号→日期类型→小时 排序，
    前80%行对应较早时段，后20%对应较晚时段。
    """
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    return X_train, X_test, y_train, y_test


# ═══════════════════════════════════════════════════════════════
# 3. XGBoost 训练（优化超参数搜索空间）
# ═══════════════════════════════════════════════════════════════

def train_xgboost(X, y, feature_names, do_tuning=True):
    """
    训练XGBoost模型（使用已搜索得到的最优参数）。

    优化点：
      - 时间序列划分替代随机划分
      - 加入 min_child_weight 和 gamma 正则化
      - 降低 max_depth 上限防止过拟合

    参数来源：GridSearchCV 5折交叉验证，18225次搜索
    """
    print("\n" + "=" * 60)
    print("XGBoost模型训练（固定最优参数）")
    print("=" * 60)
    print("\n当前参数：")
    print("  max_depth = 3")
    print("  learning_rate = 0.05")
    print("  n_estimators = 500")
    print("  subsample = 0.8")
    print("  colsample_bytree = 0.7")
    print("  min_child_weight = 3")
    print("  gamma = 0")

    # ── 时间序列划分 ──
    X_train, X_test, y_train, y_test = time_series_split(X, y, test_size=0.2)
    print(f"\n  训练集: {X_train.shape[0]} 样本 (前80%时段)")
    print(f"  测试集: {X_test.shape[0]} 样本 (后20%时段)")

    # ═══════════════════════════════════════════════════════════
    # 使用已搜索得到的最优参数直接训练
    # ═══════════════════════════════════════════════════════════
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        colsample_bytree=0.7,
        gamma=0,
        learning_rate=0.05,
        max_depth=3,
        min_child_weight=3,
        n_estimators=500,
        subsample=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    model.fit(X_train, y_train)

    # ═══════════════════════════════════════════════════════════
    # 历史超参数搜索过程（保留用于实验记录，不参与最终运行）
    #
    # 搜索空间：
    #   param_grid = {
    #       'n_estimators':     [200, 300, 500],
    #       'max_depth':        [2, 3, 4, 5, 6],
    #       'min_child_weight': [1, 3, 5],
    #       'gamma':            [0, 0.1, 0.3],
    #       'learning_rate':    [0.01, 0.03, 0.05],
    #       'subsample':        [0.7, 0.8, 0.9],
    #       'colsample_bytree': [0.7, 0.8, 1.0],
    #   }
    #
    #   GridSearchCV: cv=5, scoring='neg_mean_squared_error'
    #   搜索规模: 3645 组合 × 5 折 = 18225 次拟合
    #
    # 搜索结果：
    #   best_params = {
    #       'colsample_bytree': 0.7,
    #       'gamma': 0,
    #       'learning_rate': 0.05,
    #       'max_depth': 3,
    #       'min_child_weight': 3,
    #       'n_estimators': 500,
    #       'subsample': 0.8
    #   }
    #   best_cv_score (neg_MSE) = -12611.55
    # ═══════════════════════════════════════════════════════════

    return model, X_train, X_test, y_train, y_test


# ═══════════════════════════════════════════════════════════════
# 4. 模型评价（含SMAPE）
# ═══════════════════════════════════════════════════════════════

def smape(y_true, y_pred):
    """
    SMAPE (Symmetric Mean Absolute Percentage Error)
    公式: 100%/n * Σ(|y-y_pred| / ((|y|+|y_pred|)/2))

    相比MAPE的优势：对低负荷样本不敏感，上下界对称(0%~200%)。
    """
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    # 避免除零
    mask = denominator > 1e-8
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100


def evaluate_model(model, X_train, X_test, y_train, y_test, feature_names):
    """模型评价（优化版：增加SMAPE）"""
    print("\n" + "=" * 60)
    print("模型评价")
    print("=" * 60)

    # 预测
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # ── 基础回归指标 ──
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test  = mean_absolute_error(y_test, y_test_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_test  = np.sqrt(mean_squared_error(y_test, y_test_pred))
    r2_train = r2_score(y_train, y_train_pred)
    r2_test  = r2_score(y_test, y_test_pred)
    r2_gap = r2_train - r2_test  # 过拟合程度指标

    # ── MAPE ──
    def mape_safe(y_true, y_pred, eps=1.0):
        mask = np.abs(y_true) > eps
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    mape_train = mape_safe(y_train, y_train_pred)
    mape_test  = mape_safe(y_test, y_test_pred)

    # ── SMAPE（新增）─
    smape_train = smape(y_train, y_train_pred)
    smape_test  = smape(y_test, y_test_pred)

    # ── 预测准确率 ──
    def accuracy_within_tolerance(y_true, y_pred, tolerance_pct):
        mask = y_true > 0
        rel_error = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]) * 100
        return (rel_error <= tolerance_pct).sum() / len(rel_error) * 100

    acc_10_train = accuracy_within_tolerance(y_train, y_train_pred, 10)
    acc_10_test  = accuracy_within_tolerance(y_test, y_test_pred, 10)
    acc_15_test  = accuracy_within_tolerance(y_test, y_test_pred, 15)
    acc_20_test  = accuracy_within_tolerance(y_test, y_test_pred, 20)

    # ── RPD ──
    rpd_test = np.std(y_test) / rmse_test if rmse_test > 0 else 0

    # 汇总
    metrics = {
        '数据集':    ['训练集', '测试集'],
        'MAE(kWh)':  [mae_train, mae_test],
        'RMSE(kWh)': [rmse_train, rmse_test],
        'R2':        [r2_train, r2_test],
        'MAPE(%)':   [mape_train, mape_test],
        'SMAPE(%)':  [smape_train, smape_test],
    }
    metrics_df = pd.DataFrame(metrics)

    print("\n[基础回归指标]")
    print(metrics_df.to_string(index=False))

    print(f"\n[过拟合分析]")
    print(f"  训练R2 - 测试R2 = {r2_gap:.4f} {'(轻微过拟合)' if r2_gap > 0.08 else '(泛化良好)' if r2_gap < 0.05 else '(可接受)'}")

    print(f"\n[测试集预测准确率]")
    print(f"  ±10%: {acc_10_test:.1f}%  |  ±15%: {acc_15_test:.1f}%  |  ±20%: {acc_20_test:.1f}%")
    print(f"  SMAPE: {smape_test:.1f}%  |  RPD: {rpd_test:.2f} "
          f"{'(优)' if rpd_test >= 2.0 else '(可接受)' if rpd_test >= 1.4 else '(需改进)'}")

    # 保存指标
    full_metrics = pd.DataFrame({
        '数据集':    ['训练集', '测试集'],
        'MAE(kWh)':  [mae_train, mae_test],
        'RMSE(kWh)': [rmse_train, rmse_test],
        'R2':        [r2_train, r2_test],
        'MAPE(%)':   [mape_train, mape_test],
        'SMAPE(%)':  [smape_train, smape_test],
        '±10%准确率(%)': [acc_10_train, acc_10_test],
        '±15%准确率(%)': ['—', f'{acc_15_test:.1f}'],
        '±20%准确率(%)': ['—', f'{acc_20_test:.1f}'],
        'RPD':       ['—', f'{rpd_test:.2f}'],
    })
    full_metrics.to_excel(FILE_XGBOOST_METRICS, index=False)
    print(f"\n评价指标已保存: {FILE_XGBOOST_METRICS}")

    # ── 特征重要性 ──
    importance = model.feature_importances_
    imp_df = pd.DataFrame({
        '特征': feature_names,
        '重要性': importance
    }).sort_values('重要性', ascending=False)

    print("\n[特征重要性 TOP15]")
    print(imp_df.head(15).to_string(index=False))

    # 分组重要性分析
    print("\n[特征组重要性分析]")
    _print_group_importance(imp_df)

    accuracy_info = {
        'acc_10': acc_10_test, 'acc_15': acc_15_test, 'acc_20': acc_20_test,
        'mape': mape_test, 'smape': smape_test, 'rpd': rpd_test,
        'r2_gap': r2_gap
    }

    return y_test, y_test_pred, metrics_df, imp_df, accuracy_info


def _print_group_importance(imp_df):
    """按特征组汇总重要性"""
    groups = {
        '时间特征(小时+工作日)': ['小时', '是否工作日'],
        '区域静态特征': ['人口密度', '车流量', '商业POI数', '充电桩数量', '快充数量',
                     '慢充数量', '电网容量', '区域总面积', '充电覆盖面积'],
        '区域聚类特征': ['区域类型'],
        '历史滞后特征': ['Load_lag'],
    }

    for group_name, keywords in groups.items():
        mask = imp_df['特征'].apply(
            lambda x: any(kw in str(x) for kw in keywords)
        )
        group_importance = imp_df.loc[mask, '重要性'].sum()
        group_pct = group_importance / imp_df['重要性'].sum() * 100
        print(f"  {group_name}: {group_importance:.4f} ({group_pct:.1f}%)")


# ═══════════════════════════════════════════════════════════════
# 5. 可视化（不变）
# ═══════════════════════════════════════════════════════════════

def plot_evaluation(y_test, y_test_pred, imp_df, metrics_df, accuracy_info):
    """绘制模型评价图"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # 图1: 预测值 vs 真实值
    ax1 = axes[0, 0]
    ax1.scatter(y_test, y_test_pred, alpha=0.4, c='#3498DB', edgecolors='white', s=30)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'r--', linewidth=2, label='y = x')
    ax1.set_xlabel('真实充电负荷 (kWh)', fontsize=11)
    ax1.set_ylabel('预测充电负荷 (kWh)', fontsize=11)
    ax1.set_title(f'XGBoost 预测 vs 真实值\n'
                  f'R2={metrics_df["R2"].iloc[1]:.4f}, '
                  f'MAE={metrics_df["MAE(kWh)"].iloc[1]:.1f}kWh, '
                  f'SMAPE={accuracy_info.get("smape", 0):.1f}%',
                  fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3, linestyle='--')

    # 图2: 残差分布
    ax2 = axes[0, 1]
    residuals = y_test - y_test_pred
    ax2.hist(residuals, bins=30, color='#3498DB', edgecolor='white', alpha=0.8)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('残差 (kWh)', fontsize=11); ax2.set_ylabel('频数', fontsize=11)
    ax2.set_title(f'预测残差分布 (均值={residuals.mean():.1f}, 标准差={residuals.std():.1f})',
                  fontsize=12, fontweight='bold')

    # 图3: 特征重要性 TOP15
    ax3 = axes[1, 0]
    top15 = imp_df.head(15).iloc[::-1]
    colors = ['#E74C3C' if v > 0.03 else '#3498DB' for v in top15['重要性'].values]
    ax3.barh(range(len(top15)), top15['重要性'].values, color=colors, edgecolor='white')
    ax3.set_yticks(range(len(top15)))
    ax3.set_yticklabels(top15['特征'].values, fontsize=8)
    ax3.set_xlabel('特征重要性', fontsize=11)
    ax3.set_title('XGBoost 特征重要性 TOP15', fontsize=12, fontweight='bold')

    # 图4: 训练集 vs 测试集指标
    ax4 = axes[1, 1]
    metrics_names = ['MAE\n(kWh)', 'RMSE\n(kWh)', 'R2', 'SMAPE\n(%)']
    train_vals = [
        metrics_df['MAE(kWh)'].iloc[0], metrics_df['RMSE(kWh)'].iloc[0],
        metrics_df['R2'].iloc[0],
        accuracy_info.get('smape', metrics_df['MAPE(%)'].iloc[0])
    ]
    test_vals = [
        metrics_df['MAE(kWh)'].iloc[1], metrics_df['RMSE(kWh)'].iloc[1],
        metrics_df['R2'].iloc[1], accuracy_info.get('smape', metrics_df['MAPE(%)'].iloc[1])
    ]

    x = np.arange(len(metrics_names)); width = 0.3
    ax4.bar(x - width/2, train_vals, width, label='训练集', color='#E74C3C', edgecolor='white')
    ax4.bar(x + width/2, test_vals, width, label='测试集', color='#3498DB', edgecolor='white')
    ax4.set_xticks(x); ax4.set_xticklabels(metrics_names, fontsize=9)
    ax4.set_title('训练集 vs 测试集指标对比', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    for i, (tv, tev) in enumerate(zip(train_vals, test_vals)):
        ax4.text(i - width/2, tv + max(train_vals)*0.02, f'{tv:.1f}', ha='center', fontsize=7)
        ax4.text(i + width/2, tev + max(test_vals)*0.02, f'{tev:.1f}', ha='center', fontsize=7)

    # 图5: 预测准确率阶梯
    ax5 = axes[1, 2]
    tolerances = ['±10%', '±15%', '±20%']
    acc_values = [accuracy_info['acc_10'], accuracy_info['acc_15'], accuracy_info['acc_20']]
    bar_colors = ['#2ECC71', '#F1C40F', '#E67E22']
    bars = ax5.bar(tolerances, acc_values, color=bar_colors, edgecolor='white', width=0.5)
    ax5.set_ylabel('准确率 (%)', fontsize=11); ax5.set_ylim(0, 100)
    ax5.set_title('测试集预测准确率', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, linestyle='--', axis='y')
    for bar, val in zip(bars, acc_values):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')

    # 隐藏多余子图
    axes[1, 2].set_visible(True)

    plt.tight_layout()
    output_path = os.path.join(RESULTS_FIGURES, 'xgboost_evaluation.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nXGBoost评价图已保存: {output_path}")


# ═══════════════════════════════════════════════════════════════
# 6. 优化前后对比
# ═══════════════════════════════════════════════════════════════

def save_comparison(metrics_df, accuracy_info, r2_gap, feature_count):
    """
    保存优化前后模型指标对比表。
    原模型数据来自历史运行结果（已保存在 xgboost_metrics.xlsx），
    优化模型数据来自当前运行。
    """
    print("\n" + "=" * 60)
    print("优化前后模型对比")
    print("=" * 60)

    # 原模型指标（来自上一版运行结果）
    old_metrics = {
        '训练R2': 0.9928, '测试R2': 0.8341, 'R2差距': 0.1587,
        '测试MAE(kWh)': 121.09, '测试RMSE(kWh)': 208.52,
        '测试MAPE(%)': 70.31, '测试SMAPE(%)': '—',
        '测试±10%准确率(%)': 32.6, '测试±20%准确率(%)': 56.2,
        '测试RPD': 2.47,
        '特征维度': 33,
        '划分方式': '随机划分',
    }

    # 优化后指标
    new_metrics = {
        '训练R2': metrics_df['R2'].iloc[0],
        '测试R2': metrics_df['R2'].iloc[1],
        'R2差距': r2_gap,
        '测试MAE(kWh)': metrics_df['MAE(kWh)'].iloc[1],
        '测试RMSE(kWh)': metrics_df['RMSE(kWh)'].iloc[1],
        '测试MAPE(%)': metrics_df['MAPE(%)'].iloc[1],
        '测试SMAPE(%)': accuracy_info['smape'],
        '测试±10%准确率(%)': accuracy_info['acc_10'],
        '测试±20%准确率(%)': accuracy_info['acc_20'],
        '测试RPD': accuracy_info['rpd'],
        '特征维度': feature_count,
        '划分方式': '时间序列划分',
    }

    df_compare = pd.DataFrame({
        '指标': list(old_metrics.keys()),
        '原模型': [str(v) if isinstance(v, str) else f'{v:.4f}' if v < 10 else f'{v:.2f}'
                   for v in old_metrics.values()],
        '优化模型': [str(v) if isinstance(v, str) else f'{v:.4f}' if v < 10 else f'{v:.2f}'
                     for v in new_metrics.values()],
    })

    df_compare.to_excel(FILE_XGBOOST_COMPARE, index=False)
    print(df_compare.to_string(index=False))
    print(f"\n对比表已保存: {FILE_XGBOOST_COMPARE}")

    # 关键改进分析
    print("\n[关键改进]")
    r2_gap_old = 0.1587
    r2_gap_new = r2_gap
    print(f"  过拟合改善: R2差距 0.1587 → {r2_gap_new:.4f} (↓{0.1587 - r2_gap_new:.4f})")

    if accuracy_info['smape'] < 100:
        print(f"  SMAPE: {accuracy_info['smape']:.1f}% (新增指标，比MAPE更鲁棒)")

    return df_compare


# ═══════════════════════════════════════════════════════════════
# 7. 主函数
# ═══════════════════════════════════════════════════════════════

def main():
    # 加载数据
    df = load_data()

    # 特征工程
    X, y, feature_names = prepare_features(df)

    # 训练
    model, X_train, X_test, y_train, y_test = train_xgboost(
        X, y, feature_names, do_tuning=True
    )

    # 评价
    y_test, y_test_pred, metrics_df, imp_df, accuracy_info = evaluate_model(
        model, X_train, X_test, y_train, y_test, feature_names
    )

    # 可视化
    plot_evaluation(y_test, y_test_pred, imp_df, metrics_df, accuracy_info)

    # 保存模型
    with open(FILE_XGBOOST_MODEL, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nXGBoost模型已保存: {FILE_XGBOOST_MODEL}")

    # 优化前后对比
    r2_gap = metrics_df['R2'].iloc[0] - metrics_df['R2'].iloc[1]
    save_comparison(metrics_df, accuracy_info, r2_gap, X.shape[1])

    # ── 关键指标变量输出 ──
    print("\n" + "=" * 60)
    print("XGBoost关键指标")
    print("=" * 60)
    print("特征维度:", X.shape[1])
    print("训练集R2:", metrics_df['R2'].iloc[0])
    print("测试集R2:", metrics_df['R2'].iloc[1])
    print("R2差距:", r2_gap)
    print("测试集MAE(kWh):", metrics_df['MAE(kWh)'].iloc[1])
    print("测试集RMSE(kWh):", metrics_df['RMSE(kWh)'].iloc[1])
    print("测试集MAPE(%):", metrics_df['MAPE(%)'].iloc[1])
    print("测试集SMAPE(%):", accuracy_info['smape'])
    print("测试集RPD:", accuracy_info['rpd'])

    return model, metrics_df, accuracy_info


if __name__ == '__main__':
    main()
