"""
=============================================================================
模型消融实验 — 对比不同特征/结构组合的预测性能
=============================================================================
比较5种模型变体，量化各组件贡献：
  1. Baseline XGBoost  : 仅空间特征直接预测负荷
  2. 无时间特征         : 双层模型去掉hour_sin/hour_cos/peak/weekday
  3. 无空间特征         : 双层模型仅用时间特征
  4. 单层XGBoost        : 仅XGBoost直接预测（无物理基准层）
  5. 双层模型(完整)      : LOO物理基准 + XGBoost残差

输出: results/tables/模型消融实验.xlsx, results/figures/ablation.png
=============================================================================
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

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from utils.paths import (
    RESULTS_TABLES, RESULTS_FIGURES,
    FILE_ATTACHMENT1, FILE_ATTACHMENT2, FILE_ATTACHMENT3, FILE_ATTACHMENT4
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

TIME_LABELS = ['00-01','01-02','02-03','03-04','04-05','05-06','06-07','07-08',
               '08-09','09-10','10-11','11-12','12-13','13-14','14-15','15-16',
               '16-17','17-18','18-19','19-20','20-21','21-22','22-23','23-00']
ALL_REGIONS = list(range(1, 11))
REGION_TYPES = {1:'老城核心区',2:'老城核心区',3:'老城核心区',4:'城市新区',5:'城市新区',
                6:'城市新区',7:'城郊/工业区',8:'城郊/工业区',9:'城郊/工业区',10:'城郊/工业区'}

os.makedirs(os.path.join(ROOT, 'src', 'analysis'), exist_ok=True)

print('=' * 60)
print('模型消融实验')
print('=' * 60)

# ====== 加载数据 ======
df1 = pd.read_excel(FILE_ATTACHMENT1).iloc[:10]
df1.columns = ['区域编号','区域总面积','充电覆盖面积','人口密度','车流量','商业POI数',
               '充电桩数量','快充数量','慢充数量','电网容量']
df1['区域编号'] = df1['区域编号'].astype(int)
df4 = pd.read_excel(FILE_ATTACHMENT4)
region_names = dict(zip(df4['区域'].astype(int), df4['区域名称']))

sessions, loads, daily_s, daily_l = {}, {}, {}, {}
for day_type, sh in [('工作日','工作日分时段充电车次数据'), ('周末','周末充电车次数据')]:
    df_s = pd.read_excel(FILE_ATTACHMENT2, sheet_name=sh)
    for _, row in df_s.iterrows():
        rid = int(row['区域']); total = 0
        for h, tl in enumerate(TIME_LABELS):
            v = float(row[tl]); sessions[(rid, h, day_type)] = v; total += v
        daily_s[(rid, day_type)] = total
for day_type, sh in [('工作日','工作日分时段充电负荷数据'), ('周末','周末充电负荷数据（修改后）')]:
    df_l = pd.read_excel(FILE_ATTACHMENT3, sheet_name=sh)
    for _, row in df_l.iterrows():
        rid = int(row['区域']); total = 0
        for h, tl in enumerate(TIME_LABELS):
            v = float(row[tl]); loads[(rid, h, day_type)] = v; total += v
        daily_l[(rid, day_type)] = total

region_info = {}
for _, row in df1.iterrows():
    rid = int(row['区域编号'])
    region_info[rid] = {c: float(row[c]) for c in ['充电桩数量','车流量','人口密度','商业POI数','电网容量']}

# ====== LOO base predictions ======
def compute_loo():
    loo_daily = {}
    for target_rid in ALL_REGIONS:
        for day_type in ['工作日','周末']:
            X_tr, y_tr = [], []
            for rid in ALL_REGIONS:
                if rid == target_rid: continue
                X_tr.append([daily_s.get((rid,day_type),0), region_info[rid]['充电桩数量'], region_info[rid]['车流量']])
                y_tr.append(daily_l.get((rid,day_type),0))
            m = Ridge(alpha=1.0).fit(np.array(X_tr), np.array(y_tr))
            pred = m.predict([[daily_s.get((target_rid,day_type),0),
                              region_info[target_rid]['充电桩数量'],
                              region_info[target_rid]['车流量']]])[0]
            loo_daily[(target_rid, day_type)] = max(pred, 100)
    return loo_daily

loo_daily = compute_loo()

# Hourly base predictions
base_hourly = {}
for target_rid in ALL_REGIONS:
    t = REGION_TYPES[target_rid]
    peers = [r for r in ALL_REGIONS if REGION_TYPES[r]==t and r!=target_rid]
    for day_type in ['工作日','周末']:
        dp = loo_daily[(target_rid, day_type)]
        uc = np.zeros(24)
        for h in range(24):
            pl = sum(loads.get((r,h,day_type),0) for r in peers)
            ps = sum(sessions.get((r,h,day_type),0) for r in peers)
            uc[h] = pl/ps if ps>0 else 0
        uc = np.convolve(uc, np.ones(3)/3, mode='same'); uc = np.maximum(uc, 0.1)
        ts = np.array([sessions.get((target_rid,h,day_type),0) for h in range(24)])
        rh = ts * uc; rt = rh.sum()
        for h in range(24):
            base_hourly[(target_rid,h,day_type)] = max(rh[h]*dp/rt,0) if rt>0 else 0

# ====== Build features ======
def build_features(use_spatial=True, use_temporal=True):
    X_list, y_list = [], []
    for rid in ALL_REGIONS:
        info = region_info[rid]; rt = REGION_TYPES[rid]
        for day_type in ['工作日','周末']:
            iw = 1 if day_type=='工作日' else 0
            for h in range(24):
                base = base_hourly.get((rid,h,day_type),0)
                actual = loads.get((rid,h,day_type),0)
                residual = actual - base
                feats = []
                if use_spatial:
                    feats += [info['充电桩数量'], info['车流量'], info['人口密度'], info['商业POI数'], info['电网容量']]
                if use_temporal:
                    feats += [np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24),
                             1 if 7<=h<=9 else 0, 1 if 17<=h<=20 else 0, iw]
                feats += [1 if rt=='老城核心区' else 0, 1 if rt=='城市新区' else 0, 1 if rt=='城郊/工业区' else 0]
                X_list.append(feats); y_list.append(residual)
    return np.array(X_list), np.array(y_list)

# ====== Train & evaluate ======
def evaluate_model(name, X, y, has_base=True):
    np.random.seed(42)
    n = len(X); idx = np.random.permutation(n); split = int(n*0.8)
    Xtr, Xte = X[idx[:split]], X[idx[split:]]
    ytr, yte = y[idx[:split]], y[idx[split:]]
    model = xgb.XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5,
                             reg_lambda=1.0, random_state=42, n_jobs=-1, verbosity=0)
    model.fit(Xtr, ytr)
    yp = model.predict(Xte)
    mae = mean_absolute_error(yte, yp)
    rmse = np.sqrt(mean_squared_error(yte, yp))
    r2 = r2_score(yte, yp)
    print(f'  {name:<20s}: MAE={mae:6.1f}kW, RMSE={rmse:6.1f}kW, R²={r2:.4f}')
    return {'模型': name, 'MAE(kW)': round(mae,2), 'RMSE(kW)': round(rmse,2), 'R²': round(r2,4)}

results = []

# 1. Baseline: XGBoost direct (no base, no residual)
X_full, y_full = build_features(True, True)
y_direct = np.array([loads.get((rid,h,day_type),0) for rid in ALL_REGIONS
                     for day_type in ['工作日','周末'] for h in range(24)])
results.append(evaluate_model('1_Baseline_XGBoost', X_full, y_direct, False))

# 2. No temporal features
X_nospatial, _ = build_features(True, False)
_, y_res = build_features(True, True)
results.append(evaluate_model('2_无时间特征', X_nospatial, y_res, True))

# 3. No spatial features
X_notemporal, _ = build_features(False, True)
results.append(evaluate_model('3_无空间特征', X_notemporal, y_res, True))

# 4. Single-layer XGBoost (predict residual without LOO base)
# Use the same features but target = load (no base subtraction)
results.append(evaluate_model('4_单层XGBoost', X_full, y_direct, False))

# 5. Full two-layer model
results.append(evaluate_model('5_双层模型(完整)', X_full, y_res, True))

# ====== Save ======
df_abl = pd.DataFrame(results)
df_abl.to_excel(os.path.join(RESULTS_TABLES, '模型消融实验.xlsx'), index=False)
print(f'\n✅ 消融实验保存: {os.path.join(RESULTS_TABLES, "模型消融实验.xlsx")}')

# ====== Plot ======
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics = ['MAE(kW)', 'RMSE(kW)', 'R²']
for ax, metric in zip(axes, metrics):
    vals = [r[metric] for r in results]
    names_short = [r['模型'].replace('1_','').replace('2_','').replace('3_','').replace('4_','').replace('5_','') for r in results]
    colors = ['#95A5A6','#95A5A6','#95A5A6','#95A5A6','#E74C3C']
    ax.barh(range(len(vals)), vals, color=colors, edgecolor='white')
    ax.set_yticks(range(len(vals))); ax.set_yticklabels(names_short, fontsize=9)
    ax.set_title(metric, fontweight='bold')
    if metric == 'R²': ax.set_xlabel('越高越好 →')
    else: ax.set_xlabel('越低越好 →')
    ax.invert_yaxis(); ax.grid(axis='x', alpha=0.3, ls='--')
fig.suptitle('模型消融实验 — 各组件贡献分析', fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(RESULTS_FIGURES, 'ablation.png'), dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f'✅ 消融实验图: {os.path.join(RESULTS_FIGURES, "ablation.png")}')
print('=' * 60)
print('消融实验完成！')
