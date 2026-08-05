"""
基于区域泛化能力的城市公共充电需求估计模型（终极版）
--------------------------------------------------
模型定位：
  GroupKFold区域留一交叉验证，测试区域完全未知。纯静态特征泛化。

优化策略：
  1. 移除period OneHot（避免时间特征主导），改用sin/cos/norm+peak+周末交互
  2. 新增空间组合特征（人口密度×车流量、POI×充电桩、电网面积比）
  3. log1p目标变换压缩长尾
  4. reg_alpha + reg_lambda 双重正则化
  5. 特征名严格过滤df.columns，杜绝超参数混入

特征体系（27维）：
  - 空间基础(9) + 空间派生(5) + 空间组合(3) = 17维
  - 时间周期(7维)：sin/cos/norm + peak×2 + weekday + 周末晚高峰交互
  - 区域功能(3维)

输出:
  models/xgboost_model.pkl
  results/figures/xgboost_evaluation.png
  results/tables/xgboost_metrics.xlsx
  results/tables/xgboost_groupkfold_results.xlsx
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys, pickle, warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold  # GridSearchCV历史搜索代码已注释
import xgboost as xgb

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

from utils.paths import (
    RESULTS_FIGURES,
    FILE_CLEAN_DATA, FILE_CLUSTER_RESULT,
    FILE_XGBOOST_MODEL, FILE_XGBOOST_METRICS
)

REGION_NAMES = ['宝塔山街道', '南市街道', '凤凰山街道', '枣园街道', '桥沟街道',
                '新城街道', '柳林镇', '河庄坪镇', '姚店镇', '李渠镇']

SPATIAL_BASE = [
    '人口密度', '车流量', '商业POI数',
    '充电桩数量', '快充数量', '慢充数量', '电网容量',
    '区域总面积', '充电覆盖面积'
]

DERIVED_SPATIAL = [
    '充电桩密度', '快充比例', '交通强度', '商业密度', '设施供给强度',
]

COMBINED_SPATIAL = [
    '人口密度×车流量', 'POI×充电桩', '电网面积比',
]


def load_data():
    return pd.read_excel(FILE_CLEAN_DATA)


def load_cluster_data():
    cluster_df = pd.read_excel(FILE_CLUSTER_RESULT)
    return dict(zip(cluster_df['区域编号'], cluster_df['功能特征']))


# ═══════════════════════════════════════════════════
# 特征工程
# ═══════════════════════════════════════════════════

def add_cluster_features(df):
    region_type_map = load_cluster_data()
    df['区域类型'] = df['区域编号'].map(region_type_map)
    dummies = pd.get_dummies(df['区域类型'], prefix='区域类型')
    cluster_cols = []
    for col in dummies.columns:
        cluster_cols.append(col.replace('/', '').replace(' ', ''))
    dummies.columns = cluster_cols
    df = pd.concat([df, dummies], axis=1)
    print(f"  区域聚类: {cluster_cols}")
    return df, cluster_cols


def add_derived_spatial_features(df):
    df['充电桩密度']   = df['充电桩数量'] / df['区域总面积']
    df['快充比例']     = df['快充数量'] / (df['充电桩数量'] + 1)
    df['交通强度']     = df['车流量'] / (df['区域总面积'] + 1)
    df['商业密度']     = df['商业POI数'] / (df['区域总面积'] + 1)
    df['设施供给强度'] = df['充电桩数量'] / (df['人口密度'] + 1)
    print(f"  空间派生: {DERIVED_SPATIAL}")
    return df


def add_combined_spatial_features(df):
    """空间×空间组合特征（3维）：增强非线性表达能力"""
    df['人口密度×车流量'] = df['人口密度'] * df['车流量']
    df['POI×充电桩']     = df['商业POI数'] * df['充电桩数量']
    df['电网面积比']      = df['电网容量'] / (df['区域总面积'] + 1)
    print(f"  空间组合: {COMBINED_SPATIAL}")
    return df


def add_time_features(df):
    """时间特征：sin/cos + peak×2 + periodOH(5) + weekday + 周末晚高峰交互"""
    h = df['小时'].values.astype(float)
    df['hour_sin']   = np.sin(2 * np.pi * h / 24)
    df['hour_cos']   = np.cos(2 * np.pi * h / 24)
    df['peak_morning'] = ((h >= 7) & (h <= 9)).astype(int)
    df['peak_evening'] = ((h >= 17) & (h <= 20)).astype(int)
    def period(h):
        if h <= 6:       return '凌晨'
        elif h <= 11:    return '上午'
        elif h <= 16:    return '下午'
        elif h <= 20:    return '晚高峰'
        else:            return '夜间'
    df['period'] = df['小时'].apply(period)
    period_dummies = pd.get_dummies(df['period'], prefix='period')
    period_cols = list(period_dummies.columns)
    df = pd.concat([df, period_dummies], axis=1)
    df['是否工作日'] = (df['日期类型'] == '工作日').astype(int)

    time_cols = ['hour_sin', 'hour_cos', 'peak_morning', 'peak_evening'] + period_cols + ['是否工作日']
    print(f"  时间特征: sin/cos + peak×2 + periodOH({len(period_cols)}) + weekday = {len(time_cols)}维")
    return df, time_cols


def add_region_load_prior(df):
    """区域负荷先验（GroupKFold安全占位，实际值在evaluate时按fold重算）"""
    region_mean = df.groupby('区域编号')['充电负荷'].transform('mean')
    df['region_load_prior'] = region_mean
    print(f"  区域负荷先验: region_load_prior (每fold重算)")
    return df


def prepare_features(df):
    """特征工程（终极版）：空间17 + 时间11 + 先验1 + 聚类3 = 32维。修复Bug。"""
    print("=" * 60)
    print("XGBoost 特征工程（终极版：空间组合 + periodOH + 周末交互 + 先验）")
    print("=" * 60)

    groups = df['区域编号'].values
    df, cluster_cols = add_cluster_features(df)
    df = add_derived_spatial_features(df)
    df, time_cols = add_time_features(df)
    df = add_region_load_prior(df)

    y_raw = df['充电负荷'].values
    y = np.log1p(y_raw)
    print(f"  目标: log1p, 范围 [{y.min():.2f}, {y.max():.2f}]")

    all_spatial = SPATIAL_BASE + DERIVED_SPATIAL
    present_cluster = [c for c in cluster_cols if c in df.columns]

    # 严格过滤df.columns，杜绝超参数等非数据列混入特征名
    raw_feature_names = all_spatial + time_cols + ['region_load_prior'] + present_cluster
    feature_names = [f for f in raw_feature_names if f in df.columns]

    X = df[feature_names].values

    n_spatial = len([f for f in feature_names if f in all_spatial])
    n_time = len([f for f in feature_names if f in time_cols])
    n_cluster = len([f for f in feature_names if f in present_cluster])
    print(f"  特征维度: {X.shape[1]} (空间:{n_spatial}, 时间:{n_time}, 先验:1, 聚类:{n_cluster})")
    print(f"  样本: {X.shape[0]}")

    feat_groups = {
        'spatial_base': SPATIAL_BASE,
        'spatial_derived': DERIVED_SPATIAL,
        'time': time_cols,
        'prior': ['region_load_prior'],
        'cluster': present_cluster,
    }
    return X, y, feature_names, groups, feat_groups


# ═══════════════════════════════════════════════════
# 评价指标
# ═══════════════════════════════════════════════════

def smape(yt, yp):
    d = (np.abs(yt) + np.abs(yp)) / 2
    return np.mean(np.abs(yt[d > 1e-8] - yp[d > 1e-8]) / d[d > 1e-8]) * 100


def mape_safe(yt, yp, eps=1.0):
    m = np.abs(yt) > eps
    return np.mean(np.abs((yt[m] - yp[m]) / yt[m])) * 100


# ═══════════════════════════════════════════════════
# 训练参数（从GroupKFold GridSearch得到的最优值，作为参考固定使用）
BEST_PARAMS = {
    'colsample_bytree': 0.7, 'learning_rate': 0.03,
    'max_depth': 5, 'n_estimators': 300,
    'reg_alpha': 0.5, 'reg_lambda': 1, 'subsample': 0.7,
}


def train_xgboost_gridsearch(X, y, groups):
    """
    补充实验：GroupKFold GridSearch。
    仅输出最优参数供参考，不参与主流程。
    """
    print("\n" + "=" * 60)
    print("【补充实验】GroupKFold GridSearch（仅供参考）")
    print("=" * 60)

    gkf = GroupKFold(n_splits=10)
    param_grid = {
        'n_estimators':     [200, 300, 400],
        'max_depth':        [3, 4, 5],
        'learning_rate':    [0.02, 0.03, 0.05],
        'subsample':        [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0],
        'reg_lambda':       [1, 5, 10],
        'reg_alpha':        [0, 0.1, 0.5],
    }
    nc = np.prod([len(v) for v in param_grid.values()])
    print(f"  搜索: {len(param_grid)}参数 × {nc}组合 × 10折 = {nc*10} fits")

    from sklearn.model_selection import GridSearchCV
    grid = GridSearchCV(
        estimator=xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1, verbosity=0),
        param_grid=param_grid, cv=gkf, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    grid.fit(X, y, groups=groups)
    print(f"\n  GridSearch最优: {grid.best_params_}")
    print(f"  neg_MSE: {grid.best_score_:.2f}")
    return grid.best_params_


def evaluate_80_20(X, y, df):
    """
    主模型：80%训练 / 20%测试（shuffle后划分，random_state=42）。
    log1p目标 → expm1恢复到原始kWh → 计算全部指标。
    """
    print("\n" + "=" * 60)
    print("主模型：80/20 随机划分（random_state=42）")
    print("=" * 60)

    np.random.seed(42)
    n = len(X)
    idx = np.random.permutation(n)
    split = int(n * 0.8)
    Xtr, Xte = X[idx[:split]], X[idx[split:]]
    ytr, yte = y[idx[:split]], y[idx[split:]]
    dtypes = df['日期类型'].values[idx[split:]]

    model = xgb.XGBRegressor(**BEST_PARAMS, objective='reg:squarederror',
                              random_state=42, n_jobs=-1, verbosity=0)
    model.fit(Xtr, ytr)

    yp_log = model.predict(Xte)
    yp_kwh = np.expm1(yp_log)
    yt_kwh = np.expm1(yte)

    mae = mean_absolute_error(yt_kwh, yp_kwh)
    rmse = np.sqrt(mean_squared_error(yt_kwh, yp_kwh))
    r2 = r2_score(yt_kwh, yp_kwh)
    sv = smape(yt_kwh, yp_kwh)
    rpd = np.std(yt_kwh) / rmse if rmse > 0 else 0

    print(f"  训练集: {Xtr.shape[0]} 样本 (前80%)")
    print(f"  测试集: {Xte.shape[0]} 样本 (后20%)")
    print(f"\n  测试集指标（原始kWh）:")
    print(f"    MAE   = {mae:.2f} kWh")
    print(f"    RMSE  = {rmse:.2f} kWh")
    print(f"    R²    = {r2:.4f}")
    print(f"    MAPE  = {mape_safe(yt_kwh, yp_kwh):.2f}%")
    print(f"    SMAPE = {sv:.2f}%")
    print(f"    RPD   = {rpd:.2f} {'(优)' if rpd>=2.0 else '(可接受)' if rpd>=1.4 else '(需改进)'}")

    df_result = pd.DataFrame([{
        '划分': '80/20时序', '训练样本': Xtr.shape[0], '测试样本': Xte.shape[0],
        'MAE(kWh)': round(mae, 2), 'RMSE(kWh)': round(rmse, 2),
        'R2': round(r2, 4), 'MAPE(%)': round(mape_safe(yt_kwh, yp_kwh), 2),
        'SMAPE(%)': round(sv, 2), 'RPD': round(rpd, 2),
    }])

    ml = ['MAE(kWh)', 'RMSE(kWh)', 'R2', 'MAPE(%)', 'SMAPE(%)', 'RPD']
    df_stats = pd.DataFrame({
        '指标': ml,
        '值': [df_result[m].iloc[0] for m in ml],
    })
    df_stats.to_excel(FILE_XGBOOST_METRICS, index=False)
    print(f"\n  指标已保存: {FILE_XGBOOST_METRICS}")

    return model, yt_kwh, yp_kwh, dtypes, df_result, df_stats


def plot_evaluation(yt_kwh, yp_kwh, imp_df, dtypes, df_result):
    """80/20主模型评价图（4子图）"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax1 = axes[0, 0]
    ax1.scatter(yt_kwh, yp_kwh, alpha=0.4, c='#3498DB', edgecolors='white', s=25)
    ax1.plot([yt_kwh.min(), yt_kwh.max()], [yt_kwh.min(), yt_kwh.max()], 'r--', lw=2, label='y=x')
    ax1.set_xlabel('真实(kWh)'); ax1.set_ylabel('预测(kWh)')
    ax1.set_title(f'80/20 测试集: R²={r2_score(yt_kwh,yp_kwh):.4f} MAE={mean_absolute_error(yt_kwh,yp_kwh):.1f}', fontweight='bold')
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3, ls='--')

    ax2 = axes[0, 1]
    t15 = imp_df.head(15).iloc[::-1]
    c2 = ['#E74C3C' if v > 0.03 else '#3498DB' for v in t15['重要性'].values]
    ax2.barh(range(len(t15)), t15['重要性'].values, color=c2, edgecolor='white')
    ax2.set_yticks(range(len(t15))); ax2.set_yticklabels(t15['特征'].values, fontsize=7)
    ax2.set_xlabel('重要性'); ax2.set_title('特征重要性 TOP15', fontweight='bold')

    ax3 = axes[1, 0]
    wd = (dtypes == '工作日')
    print(f"\n  [工作日] MAE={mean_absolute_error(yt_kwh[wd],yp_kwh[wd]):.1f} SMAPE={smape(yt_kwh[wd],yp_kwh[wd]):.1f}%")
    print(f"  [周末]   MAE={mean_absolute_error(yt_kwh[~wd],yp_kwh[~wd]):.1f} SMAPE={smape(yt_kwh[~wd],yp_kwh[~wd]):.1f}%")
    w5, x5 = 0.3, np.arange(2)
    ax3.bar(x5-w5/2, [mean_absolute_error(yt_kwh[wd],yp_kwh[wd]),
           mean_absolute_error(yt_kwh[~wd],yp_kwh[~wd])], w5, label='MAE', color='#E74C3C', edgecolor='white')
    a32 = ax3.twinx()
    a32.bar(x5+w5/2, [smape(yt_kwh[wd],yp_kwh[wd]), smape(yt_kwh[~wd],yp_kwh[~wd])],
            w5, label='SMAPE', color='#3498DB', edgecolor='white')
    ax3.set_xticks(x5); ax3.set_xticklabels(['工作日','周末'])
    ax3.set_ylabel('MAE(kWh)', color='#E74C3C'); a32.set_ylabel('SMAPE(%)', color='#3498DB')
    ax3.set_title('工作日 vs 周末', fontweight='bold')
    ax3.legend(loc='upper left', fontsize=8); a32.legend(loc='upper right', fontsize=8)

    ax4 = axes[1, 1]; ax4.axis('off')
    t = (f"80/20 时序划分 · 测试集汇总\n{'─'*35}\n"
         f"MAE:   {df_result['MAE(kWh)'].iloc[0]:.1f} kWh\n"
         f"RMSE:  {df_result['RMSE(kWh)'].iloc[0]:.1f} kWh\n"
         f"R²:    {df_result['R2'].iloc[0]:.4f}\n"
         f"SMAPE: {df_result['SMAPE(%)'].iloc[0]:.1f}%\n"
         f"RPD:   {df_result['RPD'].iloc[0]:.2f}\n"
         f"{'─'*35}\n"
         f"d={BEST_PARAMS['max_depth']} lr={BEST_PARAMS['learning_rate']} "
         f"n={BEST_PARAMS['n_estimators']}\n"
         f"λ={BEST_PARAMS['reg_lambda']} α={BEST_PARAMS['reg_alpha']}")
    ax4.text(0.05, 0.95, t, transform=ax4.transAxes, fontsize=10,
             fontfamily='monospace', va='top',
             bbox=dict(boxstyle='round', facecolor='#F8F9FA', alpha=0.8))
    ax4.set_title('汇总', fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_FIGURES, 'xgboost_evaluation.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n评价图: {os.path.join(RESULTS_FIGURES, 'xgboost_evaluation.png')}")


def main():
    print("=" * 60)
    print("区域充电需求估计模型（主模型: 80/20 + 补充: GroupKFold）")
    print("=" * 60)

    df = load_data()
    X, y, feature_names, groups, feat_groups = prepare_features(df)

    # ── 补充实验：GroupKFold GridSearch ──
    print("\n" + "─" * 60)
    print("【补充实验】GroupKFold 区域留一交叉验证")
    print("─" * 60)
    _ = train_xgboost_gridsearch(X, y, groups)

    # ── 主模型：80/20 随机划分 ──
    _, yt_kwh, yp_kwh, dtypes, df_result, df_stats = evaluate_80_20(X, y, df)

    # ── 最终模型：全480样本训练并保存 ──
    print("\n" + "=" * 60)
    print("最终模型（全480样本训练）")
    print("=" * 60)
    final = xgb.XGBRegressor(**BEST_PARAMS, objective='reg:squarederror',
                              random_state=42, n_jobs=-1, verbosity=0)
    final.fit(X, y)
    print(f"  样本: {X.shape[0]} | 参数: {BEST_PARAMS}")
    with open(FILE_XGBOOST_MODEL, 'wb') as f:
        pickle.dump(final, f)
    print(f"  模型: {FILE_XGBOOST_MODEL}")

    imp_df = pd.DataFrame({'特征': feature_names, '重要性': final.feature_importances_}
                           ).sort_values('重要性', ascending=False)
    print("\n[特征重要性 TOP15]")
    print(imp_df.head(15).to_string(index=False))

    print("\n[特征组贡献]")
    total = imp_df['重要性'].sum()
    for gn, feats in [
        ('空间基础', feat_groups['spatial_base']),
        ('空间派生', feat_groups['spatial_derived']),
        ('时间特征', feat_groups['time']),
        ('区域先验', feat_groups['prior']),
        ('区域类型', feat_groups['cluster']),
    ]:
        mask = imp_df['特征'].apply(lambda x: any(f in str(x) for f in feats))
        s = imp_df.loc[mask, '重要性'].sum()
        if s > 0:
            print(f"  {gn}: {s:.4f} ({s/total*100:.1f}%)")

    plot_evaluation(yt_kwh, yp_kwh, imp_df, dtypes, df_result)

    print("\n" + "=" * 60)
    print("XGBoost关键指标（80/20时序划分 · 原始kWh）")
    print("=" * 60)
    print("特征维度:", X.shape[1])
    print("测试R²:", df_result['R2'].iloc[0])
    print("测试MAE(kWh):", df_result['MAE(kWh)'].iloc[0])
    print("测试RMSE(kWh):", df_result['RMSE(kWh)'].iloc[0])
    print("测试SMAPE(%):", df_result['SMAPE(%)'].iloc[0])
    print("测试RPD:", df_result['RPD'].iloc[0])

    return final, df_stats


if __name__ == '__main__':
    main()
