"""
Step 4: XGBoost 充电需求预测模型
-------------------------------
主模型：基于多维特征预测充电负荷。
输出: model/xgboost_model.pkl
      result/figures/xgboost_evaluation.png
      result/tables/xgboost_metrics.xlsx
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

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

from utils.paths import RESULTS_DIR, RESULTS_FIGURES, RESULTS_TABLES, MODELS_DIR, FILE_CLEAN_DATA, FILE_XGBOOST_MODEL, FILE_XGBOOST_METRICS


def load_data():
    return pd.read_excel(FILE_CLEAN_DATA)


def prepare_features(df, corr_threshold=0.05):
    """特征工程 + 基于相关性分析的特征筛选"""
    print("=" * 60)
    print("XGBoost 特征工程（含相关性筛选）")
    print("=" * 60)

    # 目标变量
    y = df['充电负荷'].values

    # ── 第一步：相关性筛选连续特征 ──
    all_continuous = [
        '人口密度', '车流量', '商业POI数',
        '充电桩数量', '快充数量', '慢充数量', '电网容量',
        '区域总面积', '充电覆盖面积'
    ]

    # 计算各连续特征与目标的相关性
    corr_with_target = {}
    for feat in all_continuous:
        if feat in df.columns:
            r = df[feat].corr(df['充电负荷'])
            corr_with_target[feat] = r

    # 筛选：|r| >= threshold
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

    # ── 第二步：类别特征编码 ──
    df_feat = df.copy()
    df_feat['是否工作日'] = (df_feat['日期类型'] == '工作日').astype(int)

    # 小时 one-hot（保留全部24小时，时间特征对负荷预测至关重要）
    hour_dummies = pd.get_dummies(df_feat['小时'], prefix='小时')

    # 构建特征矩阵
    X = np.hstack([
        df_feat[selected_continuous].values,   # 筛选后的连续特征
        df_feat[['是否工作日']].values,          # 工作日标志
        hour_dummies.values,                    # 小时 one-hot
    ])

    feature_names = (
        selected_continuous +
        ['是否工作日'] +
        list(hour_dummies.columns)
    )

    print(f"\n  → 特征维度: {X.shape[1]} (连续: {len(selected_continuous)}, "
          f"类别: 1 + 24小时 = 25)")
    print(f"  → 样本数量: {X.shape[0]}")
    print(f"  → 剔除低相关特征: {len(dropped_continuous)} 个")

    return X, y, feature_names


def train_xgboost(X, y, feature_names, do_tuning=True):
    """训练XGBoost模型"""
    print("\n" + "=" * 60)
    print("XGBoost 模型训练")
    print("=" * 60)

    # 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n  训练集: {X_train.shape[0]} 样本")
    print(f"  测试集: {X_test.shape[0]} 样本")

    if do_tuning:
        # 超参数网格搜索
        print("\n[超参数调优] GridSearchCV...")
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.8, 1.0],
            'colsample_bytree': [0.7, 0.8, 1.0],
        }

        base_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )

        grid = GridSearchCV(
            base_model, param_grid,
            cv=5, scoring='neg_mean_squared_error',
            n_jobs=-1, verbose=1
        )
        grid.fit(X_train, y_train)

        model = grid.best_estimator_
        print(f"\n  最佳参数: {grid.best_params_}")
        print(f"  最佳CV分数 (neg_MSE): {grid.best_score_:.2f}")
    else:
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test


def evaluate_model(model, X_train, X_test, y_train, y_test, feature_names):
    """模型评价"""
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

    # ── MAPE: 避免除零 (忽略 y_true < 1 的样本) ──
    def mape_safe(y_true, y_pred, eps=1.0):
        mask = np.abs(y_true) > eps
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    mape_train = mape_safe(y_train, y_train_pred)
    mape_test  = mape_safe(y_test, y_test_pred)

    # ── 预测准确率: 预测值在真实值 ±T% 范围内的样本占比 ──
    def accuracy_within_tolerance(y_true, y_pred, tolerance_pct):
        """回归准确率：预测误差在 ±tolerance_pct% 以内"""
        # 仅对真实值 > 0 的样本计算相对误差
        mask = y_true > 0
        rel_error = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]) * 100
        return (rel_error <= tolerance_pct).sum() / len(rel_error) * 100

    acc_10_train = accuracy_within_tolerance(y_train, y_train_pred, 10)
    acc_10_test  = accuracy_within_tolerance(y_test, y_test_pred, 10)
    acc_15_test  = accuracy_within_tolerance(y_test, y_test_pred, 15)
    acc_20_test  = accuracy_within_tolerance(y_test, y_test_pred, 20)

    # ── RPD (Ratio of Performance to Deviation) ──
    rpd_test = np.std(y_test) / rmse_test if rmse_test > 0 else 0

    # 汇总
    metrics = {
        '数据集':   ['训练集', '测试集'],
        'MAE(kWh)': [mae_train, mae_test],
        'RMSE(kWh)':[rmse_train, rmse_test],
        'R2':       [r2_train, r2_test],
        'MAPE(%)':  [mape_train, mape_test],
        '±10%准确率(%)': [acc_10_train, acc_10_test],
    }

    metrics_df = pd.DataFrame(metrics)

    print("\n[基础回归指标]")
    print(metrics_df[['数据集', 'MAE(kWh)', 'RMSE(kWh)', 'R2', 'MAPE(%)']].to_string(index=False))

    print(f"\n[测试集预测准确率]")
    print(f"  预测误差在 ±10% 以内: {acc_10_test:.1f}%")
    print(f"  预测误差在 ±15% 以内: {acc_15_test:.1f}%")
    print(f"  预测误差在 ±20% 以内: {acc_20_test:.1f}%")
    print(f"  RPD (标准差/均方根误差): {rpd_test:.2f} {'(优)' if rpd_test >= 2.0 else '(可接受)' if rpd_test >= 1.4 else '(需改进)'}")

    # 保存指标
    full_metrics = pd.DataFrame({
        '数据集':   ['训练集', '测试集'],
        'MAE(kWh)': [mae_train, mae_test],
        'RMSE(kWh)':[rmse_train, rmse_test],
        'R2':       [r2_train, r2_test],
        'MAPE(%)':  [mape_train, mape_test],
        '±10%准确率(%)': [acc_10_train, acc_10_test],
        '±15%准确率(%)': ['—', f'{acc_15_test:.1f}'],
        '±20%准确率(%)': ['—', f'{acc_20_test:.1f}'],
        'RPD':      ['—', f'{rpd_test:.2f}'],
    })
    full_metrics.to_excel(FILE_XGBOOST_METRICS, index=False)
    print(f"\n✅ 评价指标已保存: {FILE_XGBOOST_METRICS}")

    # ── 特征重要性 ──
    importance = model.feature_importances_
    imp_df = pd.DataFrame({
        '特征': feature_names,
        '重要性': importance
    }).sort_values('重要性', ascending=False)

    print("\n[特征重要性 TOP10]")
    print(imp_df.head(10).to_string(index=False))

    # 打包额外准确率信息
    accuracy_info = {
        'acc_10': acc_10_test, 'acc_15': acc_15_test, 'acc_20': acc_20_test,
        'mape': mape_test, 'rpd': rpd_test
    }

    return y_test, y_test_pred, metrics_df, imp_df, accuracy_info


def plot_evaluation(y_test, y_test_pred, imp_df, metrics_df, accuracy_info):
    """绘制模型评价图"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # ── 图1: 预测值 vs 真实值散点图 ──
    ax1 = axes[0, 0]
    ax1.scatter(y_test, y_test_pred, alpha=0.4, c='#3498DB', edgecolors='white', s=30)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'r--', linewidth=2, label='y = x (完美预测)')
    ax1.set_xlabel('真实充电负荷 (kWh)', fontsize=11)
    ax1.set_ylabel('预测充电负荷 (kWh)', fontsize=11)
    ax1.set_title(f'XGBoost 预测 vs 真实值\n'
                  f'R2 = {metrics_df["R2"].iloc[1]:.4f}, '
                  f'MAE = {metrics_df["MAE(kWh)"].iloc[1]:.1f} kWh',
                  fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # ── 图2: 残差分布 ──
    ax2 = axes[0, 1]
    residuals = y_test - y_test_pred
    ax2.hist(residuals, bins=30, color='#3498DB', edgecolor='white', alpha=0.8)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('残差 (kWh)', fontsize=11)
    ax2.set_ylabel('频数', fontsize=11)
    ax2.set_title(f'预测残差分布 (均值={residuals.mean():.1f}, 标准差={residuals.std():.1f})',
                  fontsize=12, fontweight='bold')

    # ── 图3: 特征重要性 TOP15 ──
    ax3 = axes[1, 0]
    top15 = imp_df.head(15).iloc[::-1]
    colors = ['#E74C3C' if v > 0.03 else '#3498DB' for v in top15['重要性'].values]
    ax3.barh(range(len(top15)), top15['重要性'].values, color=colors, edgecolor='white')
    ax3.set_yticks(range(len(top15)))
    ax3.set_yticklabels(top15['特征'].values, fontsize=8)
    ax3.set_xlabel('特征重要性', fontsize=11)
    ax3.set_title('XGBoost 特征重要性 TOP15', fontsize=12, fontweight='bold')

    # ── 图4: 模型评价指标对比 ──
    ax4 = axes[1, 0]
    metrics_names = ['MAE\n(kWh)', 'RMSE\n(kWh)', 'R2', 'MAPE\n(%)']
    train_vals = [
        metrics_df['MAE(kWh)'].iloc[0], metrics_df['RMSE(kWh)'].iloc[0],
        metrics_df['R2'].iloc[0], metrics_df['MAPE(%)'].iloc[0]
    ]
    test_vals = [
        metrics_df['MAE(kWh)'].iloc[1], metrics_df['RMSE(kWh)'].iloc[1],
        metrics_df['R2'].iloc[1], metrics_df['MAPE(%)'].iloc[1]
    ]

    x = np.arange(len(metrics_names))
    width = 0.3
    ax4.bar(x - width/2, train_vals, width, label='训练集', color='#E74C3C', edgecolor='white')
    ax4.bar(x + width/2, test_vals, width, label='测试集', color='#3498DB', edgecolor='white')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics_names, fontsize=9)
    ax4.set_title('训练集 vs 测试集指标对比', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)

    for i, (tv, tev) in enumerate(zip(train_vals, test_vals)):
        ax4.text(i - width/2, tv + max(train_vals)*0.02, f'{tv:.1f}', ha='center', fontsize=7)
        ax4.text(i + width/2, tev + max(test_vals)*0.02, f'{tev:.1f}', ha='center', fontsize=7)

    # ── 图5: 测试集预测准确率阶梯图 ──
    ax5 = axes[1, 1]
    tolerances = ['±10%', '±15%', '±20%']
    acc_values = [accuracy_info['acc_10'], accuracy_info['acc_15'], accuracy_info['acc_20']]
    bar_colors = ['#2ECC71', '#F1C40F', '#E67E22']
    bars = ax5.bar(tolerances, acc_values, color=bar_colors, edgecolor='white', width=0.5)
    ax5.set_ylabel('准确率 (%)', fontsize=11)
    ax5.set_ylim(0, 100)
    ax5.set_title('测试集预测准确率 (不同容差)', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, linestyle='--', axis='y')

    for bar, val in zip(bars, acc_values):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')

    # ── 图6: 相对误差分布 + 累积准确率曲线 ──
    ax6 = axes[1, 2]
    mask = y_test > 0
    rel_errors = np.abs((y_test[mask] - y_test_pred[mask]) / y_test[mask]) * 100
    ax6.hist(rel_errors, bins=30, color='#9B59B6', edgecolor='white', alpha=0.7,
             range=(0, 50))
    ax6.axvline(x=10, color='#2ECC71', linestyle='--', linewidth=1.5, label=f'±10%: {accuracy_info["acc_10"]:.0f}%')
    ax6.axvline(x=15, color='#F1C40F', linestyle='--', linewidth=1.5, label=f'±15%: {accuracy_info["acc_15"]:.0f}%')
    ax6.axvline(x=20, color='#E67E22', linestyle='--', linewidth=1.5, label=f'±20%: {accuracy_info["acc_20"]:.0f}%')
    ax6.set_xlabel('相对误差 (%)', fontsize=11)
    ax6.set_ylabel('样本数', fontsize=11)
    ax6.set_title(f'测试集相对误差分布 (MAPE={accuracy_info["mape"]:.1f}%)',
                  fontsize=12, fontweight='bold')
    ax6.legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    output_path = os.path.join(RESULTS_FIGURES, 'xgboost_evaluation.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n✅ XGBoost评价图已保存: {output_path}")


def main():
    df = load_data()
    X, y, feature_names = prepare_features(df)
    model, X_train, X_test, y_train, y_test = train_xgboost(
        X, y, feature_names, do_tuning=True
    )
    y_test, y_test_pred, metrics_df, imp_df, accuracy_info = evaluate_model(
        model, X_train, X_test, y_train, y_test, feature_names
    )
    plot_evaluation(y_test, y_test_pred, imp_df, metrics_df, accuracy_info)

    # 保存模型
    model_path = FILE_XGBOOST_MODEL
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nXGBoost模型已保存: {model_path}")

    # ── XGBoost关键指标变量输出 ──
    test_r2 = metrics_df['R2'].iloc[1]
    test_mae = metrics_df['MAE(kWh)'].iloc[1]
    test_rmse = metrics_df['RMSE(kWh)'].iloc[1]
    test_mape = accuracy_info['mape']
    test_rpd = accuracy_info['rpd']

    print("\n" + "=" * 60)
    print("XGBoost关键指标")
    print("=" * 60)
    print("测试集R2:", test_r2)
    print("测试集MAE(kWh):", test_mae)
    print("测试集RMSE(kWh):", test_rmse)
    print("测试集MAPE(%):", test_mape)
    print("测试集RPD:", test_rpd)

    return model, metrics_df, accuracy_info


if __name__ == '__main__':
    main()
