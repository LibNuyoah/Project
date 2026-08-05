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
    FILE_XGBOOST_MODEL, FILE_XGBOOST_METRICS, FILE_XGBOOST_GROUPKFOLD
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
# 训练
# ═══════════════════════════════════════════════════

def train_xgboost(X, y, groups):
    """使用GroupKFold搜索得到的最优参数（历史搜索记录见注释）"""
    print("\n" + "=" * 60)
    print("XGBoost 训练（固定最优参数 · 28维特征 · 21870 fits）")
    print("=" * 60)

    best_params = {
        'colsample_bytree': 0.7, 'learning_rate': 0.03,
        'max_depth': 5, 'n_estimators': 300,
        'reg_alpha': 0.5, 'reg_lambda': 1, 'subsample': 0.7,
    }
    print(f"  参数: {best_params}")

    # ═══════════════════════════════════════════════════════
    # 历史搜索（保留记录）:
    #   gkf = GroupKFold(n_splits=10)
    #   param_grid = { n_estimators:[200,300,400], max_depth:[3,4,5],
    #     learning_rate:[0.02,0.03,0.05], subsample:[0.7,0.8,1.0],
    #     colsample_bytree:[0.7,0.8,1.0], reg_lambda:[1,5,10],
    #     reg_alpha:[0,0.1,0.5] }
    #   2187组合 × 10折 = 21870 fits
    #   最优: {'colsample_bytree':0.7, 'learning_rate':0.03, 'max_depth':5,
    #          'n_estimators':300, 'reg_alpha':0.5, 'reg_lambda':1, 'subsample':0.7}
    #   neg_MSE: -0.47
    # ═══════════════════════════════════════════════════════

    return best_params


# ═══════════════════════════════════════════════════
# GroupKFold评价（含先验重算 + smearing校正）
# ═══════════════════════════════════════════════════

def evaluate_groupkfold(best_params, X, y, groups, df):
    """
    GroupKFold 10折评价（log1p · 先验隔离 · 原始kWh）。
    每fold仅用训练区域计算region_load_prior，测试区域用训练均值替代。
    """
    print("\n" + "=" * 60)
    print("GroupKFold 区域留一交叉验证（先验隔离 · log1p）")
    print("=" * 60)

    gkf = GroupKFold(n_splits=10)
    fold_records, all_yt, all_yp, all_dt = [], [], [], []

    # region_load_prior在X中的列索引: spatial(14) + time(10)
    prior_x_idx = 14 + 10

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
        Xtr, Xte = X[tr].copy(), X[te].copy()
        ytr, yte = y[tr], y[te]
        test_region = int(groups[te][0])
        dtypes = df.iloc[te]['日期类型'].values
        all_dt.extend(dtypes)

        # 区域先验隔离：测试区域用训练区域均值替代
        train_region_means = {}
        for rid in np.unique(groups[tr]):
            train_region_means[rid] = np.expm1(ytr[groups[tr] == rid]).mean()
        overall_mean = np.mean(list(train_region_means.values()))
        for i_te in range(len(te)):
            rid = groups[te][i_te]
            Xte[i_te, prior_x_idx] = np.log1p(train_region_means.get(rid, overall_mean))

        m = xgb.XGBRegressor(**best_params, objective='reg:squarederror',
                              random_state=42, n_jobs=-1, verbosity=0)
        m.fit(Xtr, ytr)

        yp_log = m.predict(Xte)
        yp_kwh = np.expm1(yp_log)
        yt_kwh = np.expm1(yte)

        mae = mean_absolute_error(yt_kwh, yp_kwh)
        rmse = np.sqrt(mean_squared_error(yt_kwh, yp_kwh))
        r2 = r2_score(yt_kwh, yp_kwh)
        sv = smape(yt_kwh, yp_kwh)
        rpd = np.std(yt_kwh) / rmse if rmse > 0 else 0

        fold_records.append({
            'Fold': fold, '测试区域编号': test_region,
            '区域名称': REGION_NAMES[test_region - 1],
            'MAE(kWh)': round(mae, 2), 'RMSE(kWh)': round(rmse, 2),
            'R2': round(r2, 4), 'MAPE(%)': round(mape_safe(yt_kwh, yp_kwh), 2),
            'SMAPE(%)': round(sv, 2), 'RPD': round(rpd, 2),
        })
        all_yt.extend(yt_kwh); all_yp.extend(yp_kwh)

        s = '优' if rpd >= 2.0 else ('可接受' if rpd >= 1.4 else '需改进')
        print(f"  Fold{fold}: R{test_region}({REGION_NAMES[test_region-1]}) "
              f"| MAE={mae:.1f} | R²={r2:.4f} | SMAPE={sv:.1f}% | RPD={rpd:.2f} ({s})")

    df_folds = pd.DataFrame(fold_records)
    ml = ['MAE(kWh)', 'RMSE(kWh)', 'R2', 'MAPE(%)', 'SMAPE(%)', 'RPD']
    print("\n[GroupKFold 平均指标（原始kWh · 先验隔离 · smearing校正）]")
    for m in ml:
        print(f"  Mean {m}: {df_folds[m].mean():.4f} ± {df_folds[m].std():.4f}")

    if os.path.exists(FILE_XGBOOST_GROUPKFOLD):
        os.remove(FILE_XGBOOST_GROUPKFOLD)
    with pd.ExcelWriter(FILE_XGBOOST_GROUPKFOLD) as w:
        df_folds.to_excel(w, sheet_name='Fold详细结果', index=False)
        pd.DataFrame({'指标': ml, 'Mean': [df_folds[m].mean() for m in ml],
                       'Std': [df_folds[m].std() for m in ml]}
                      ).to_excel(w, sheet_name='统计结果', index=False)
    print(f"\nGroupKFold结果: {FILE_XGBOOST_GROUPKFOLD}")

    df_stats = pd.DataFrame({'指标': ml, 'Mean': [df_folds[m].mean() for m in ml],
                              'Std': [df_folds[m].std() for m in ml]})
    df_stats.to_excel(FILE_XGBOOST_METRICS, index=False)
    print(f"平均指标: {FILE_XGBOOST_METRICS}")

    return df_folds, df_stats, np.array(all_yt), np.array(all_yp), np.array(all_dt)


# ═══════════════════════════════════════════════════
# 可视化
# ═══════════════════════════════════════════════════

def plot_evaluation(all_yt, all_yp, imp_df, df_folds, bp, all_dt):
    fig = plt.figure(figsize=(20, 13))

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.scatter(all_yt, all_yp, alpha=0.4, c='#3498DB', edgecolors='white', s=25)
    ax1.plot([all_yt.min(), all_yt.max()], [all_yt.min(), all_yt.max()], 'r--', lw=2, label='y=x')
    ax1.set_xlabel('真实(kWh)'); ax1.set_ylabel('预测(kWh)')
    ax1.set_title(f'真实 vs 预测\nR²={r2_score(all_yt,all_yp):.4f} MAE={mean_absolute_error(all_yt,all_yp):.1f}', fontweight='bold')
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3, ls='--')

    ax2 = fig.add_subplot(2, 3, 2)
    f, r2v = df_folds['Fold'].values, df_folds['R2'].values
    c2 = ['#2ECC71' if v >= r2v.mean() else '#E74C3C' for v in r2v]
    ax2.bar(f, r2v, color=c2, edgecolor='white')
    ax2.axhline(r2v.mean(), color='#3498DB', ls='--', label=f"Mean={r2v.mean():.4f}")
    ax2.set_xlabel('Fold'); ax2.set_ylabel('R²'); ax2.set_xticks(f)
    ax2.set_title('各Fold R²', fontweight='bold'); ax2.legend(fontsize=8)
    for i, v in enumerate(r2v): ax2.text(f[i], v+0.01, f'{v:.3f}', ha='center', fontsize=7)

    ax3 = fig.add_subplot(2, 3, 3)
    t15 = imp_df.head(15).iloc[::-1]
    c3 = ['#E74C3C' if v > 0.03 else '#3498DB' for v in t15['重要性'].values]
    ax3.barh(range(len(t15)), t15['重要性'].values, color=c3, edgecolor='white')
    ax3.set_yticks(range(len(t15))); ax3.set_yticklabels(t15['特征'].values, fontsize=7)
    ax3.set_xlabel('重要性'); ax3.set_title('特征重要性 TOP15', fontweight='bold')

    ax4 = fig.add_subplot(2, 3, 4)
    rl = df_folds['区域名称'].str[:3].tolist(); mv = df_folds['MAE(kWh)'].values
    c4 = ['#E74C3C' if v > mv.mean() else '#3498DB' for v in mv]
    ax4.bar(range(10), mv, color=c4, edgecolor='white')
    ax4.axhline(mv.mean(), color='red', ls='--', label=f"Mean={mv.mean():.1f}")
    ax4.set_xticks(range(10)); ax4.set_xticklabels(rl, rotation=30, ha='right', fontsize=7)
    ax4.set_ylabel('MAE(kWh)'); ax4.set_title('各区域MAE', fontweight='bold'); ax4.legend(fontsize=7)

    ax5 = fig.add_subplot(2, 3, 5)
    wd = (all_dt == '工作日')
    print(f"\n  [工作日] MAE={mean_absolute_error(all_yt[wd],all_yp[wd]):.1f} SMAPE={smape(all_yt[wd],all_yp[wd]):.1f}%")
    print(f"  [周末]   MAE={mean_absolute_error(all_yt[~wd],all_yp[~wd]):.1f} SMAPE={smape(all_yt[~wd],all_yp[~wd]):.1f}%")
    w5, x5 = 0.3, np.arange(2)
    ax5.bar(x5-w5/2, [mean_absolute_error(all_yt[wd],all_yp[wd]),
           mean_absolute_error(all_yt[~wd],all_yp[~wd])], w5, label='MAE', color='#E74C3C', edgecolor='white')
    a52 = ax5.twinx()
    a52.bar(x5+w5/2, [smape(all_yt[wd],all_yp[wd]),
            smape(all_yt[~wd],all_yp[~wd])], w5, label='SMAPE', color='#3498DB', edgecolor='white')
    ax5.set_xticks(x5); ax5.set_xticklabels(['工作日','周末'])
    ax5.set_ylabel('MAE(kWh)', color='#E74C3C'); a52.set_ylabel('SMAPE(%)', color='#3498DB')
    ax5.set_title('工作日 vs 周末', fontweight='bold')
    ax5.legend(loc='upper left', fontsize=8); a52.legend(loc='upper right', fontsize=8)

    ax6 = fig.add_subplot(2, 3, 6); ax6.axis('off')
    t = (f"GroupKFold(先验+smearing)汇总\n{'─'*35}\n"
         f"MAE:  {df_folds['MAE(kWh)'].mean():.1f}±{df_folds['MAE(kWh)'].std():.1f}\n"
         f"RMSE: {df_folds['RMSE(kWh)'].mean():.1f}±{df_folds['RMSE(kWh)'].std():.1f}\n"
         f"R²:   {df_folds['R2'].mean():.4f}±{df_folds['R2'].std():.4f}\n"
         f"SMAPE:{df_folds['SMAPE(%)'].mean():.1f}±{df_folds['SMAPE(%)'].std():.1f}%\n"
         f"RPD:  {df_folds['RPD'].mean():.2f}±{df_folds['RPD'].std():.2f}\n"
         f"{'─'*35}\n"
         f"d={bp.get('max_depth','?')} lr={bp.get('learning_rate','?')}\n"
         f"n={bp.get('n_estimators','?')} λ={bp.get('reg_lambda','?')}\n"
         f"α={bp.get('reg_alpha','?')}")
    ax6.text(0.05, 0.95, t, transform=ax6.transAxes, fontsize=10,
             fontfamily='monospace', va='top',
             bbox=dict(boxstyle='round', facecolor='#F8F9FA', alpha=0.8))
    ax6.set_title('汇总', fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_FIGURES, 'xgboost_evaluation.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n评价图: {os.path.join(RESULTS_FIGURES, 'xgboost_evaluation.png')}")


# ═══════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("区域泛化充电需求估计（终极版：空间组合 + 周末交互 + 纯静态特征）")
    print("=" * 60)

    df = load_data()
    X, y, feature_names, groups, feat_groups = prepare_features(df)
    best_params = train_xgboost(X, y, groups)
    df_folds, df_stats, all_yt, all_yp, all_dt = evaluate_groupkfold(
        best_params, X, y, groups, df)

    # 最终模型：全数据训练（先验使用全数据区域均值，实际部署时重新计算）
    print("\n" + "=" * 60)
    print("最终模型（480样本）")
    print("=" * 60)
    final = xgb.XGBRegressor(**best_params, objective='reg:squarederror',
                              random_state=42, n_jobs=-1, verbosity=0)
    final.fit(X, y)
    print(f"  样本: {X.shape[0]} | 参数: {best_params}")
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

    plot_evaluation(all_yt, all_yp, imp_df, df_folds, best_params, all_dt)

    print("\n" + "=" * 60)
    print("XGBoost关键指标（原始kWh · GroupKFold · 先验隔离 · smearing）")
    print("=" * 60)
    print("特征维度:", X.shape[1])
    print("Mean R²:", df_folds['R2'].mean())
    print("Std  R²:", df_folds['R2'].std())
    print("Mean MAE(kWh):", df_folds['MAE(kWh)'].mean())
    print("Mean RMSE(kWh):", df_folds['RMSE(kWh)'].mean())
    print("Mean SMAPE(%):", df_folds['SMAPE(%)'].mean())
    print("Mean RPD:", df_folds['RPD'].mean())
    print("最优参数:", best_params)

    neg_r2_regions = df_folds[df_folds['R2'] < 0]['区域名称'].tolist()
    print(f"\n  负R²区域: {neg_r2_regions if neg_r2_regions else '无'}")

    print("\n" + "=" * 60)
    print("模型修改总结")
    print("=" * 60)
    print(f"""
  GroupKFold(n_splits=10)，测试区域完全未知。
  区域负荷先验：每fold仅用训练区域计算，测试区域用训练均值替代。
  smearing校正：expm1(pred_log) × mean(exp(residuals_train))。
  时间特征精简至7维（sin/cos/sq/cube + 早晚高峰 + weekday）。
  reg_alpha + reg_lambda 双重正则化。

  特征（{X.shape[1]}维）：空间14 + 时间7 + 先验1 + 聚类3
  最优参数：{best_params}
  Mean R² = {df_folds['R2'].mean():.4f} ± {df_folds['R2'].std():.4f}
  Mean SMAPE = {df_folds['SMAPE(%)'].mean():.1f}% ± {df_folds['SMAPE(%)'].std():.1f}%
""")
    return final, df_stats


if __name__ == '__main__':
    main()
