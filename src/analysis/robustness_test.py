"""
=============================================================================
鲁棒性验证 — 多随机种子重复训练评估模型稳定性
=============================================================================
使用5个不同随机种子(1,10,20,42,100)重复训练双层预测模型，
评估预测结果的稳定性和方差。

输出: results/tables/robustness.xlsx
=============================================================================
"""

import pandas as pd
import numpy as np
import os, sys, warnings
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from utils.paths import (
    RESULTS_TABLES, FILE_ATTACHMENT1, FILE_ATTACHMENT2, FILE_ATTACHMENT3, FILE_ATTACHMENT4
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
SEEDS = [1, 10, 20, 42, 100]

print('=' * 60)
print('鲁棒性验证 — 多种子重复实验')
print(f'随机种子: {SEEDS}')
print('=' * 60)

# ====== 加载数据 ======
df1 = pd.read_excel(FILE_ATTACHMENT1).iloc[:10]
df1.columns = ['区域编号','区域总面积','充电覆盖面积','人口密度','车流量','商业POI数',
               '充电桩数量','快充数量','慢充数量','电网容量']
df1['区域编号'] = df1['区域编号'].astype(int)

sessions, loads, daily_s, daily_l = {}, {}, {}, {}
for day_type, sh in [('工作日','工作日分时段充电车次数据'), ('周末','周末充电车次数据')]:
    df_s = pd.read_excel(FILE_ATTACHMENT2, sheet_name=sh)
    for _, row in df_s.iterrows():
        rid = int(row['区域']); total = 0
        for h, tl in enumerate(TIME_LABELS):
            v = float(row[tl]); sessions[(rid,h,day_type)] = v; total += v
        daily_s[(rid, day_type)] = total
for day_type, sh in [('工作日','工作日分时段充电负荷数据'), ('周末','周末充电负荷数据（修改后）')]:
    df_l = pd.read_excel(FILE_ATTACHMENT3, sheet_name=sh)
    for _, row in df_l.iterrows():
        rid = int(row['区域']); total = 0
        for h, tl in enumerate(TIME_LABELS):
            v = float(row[tl]); loads[(rid,h,day_type)] = v; total += v
        daily_l[(rid, day_type)] = total

region_info = {}
for _, row in df1.iterrows():
    rid = int(row['区域编号'])
    region_info[rid] = {c: float(row[c]) for c in ['充电桩数量','车流量','人口密度','商业POI数','电网容量']}

# LOO base
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

base_hourly = {}
for target_rid in ALL_REGIONS:
    t = REGION_TYPES[target_rid]; peers = [r for r in ALL_REGIONS if REGION_TYPES[r]==t and r!=target_rid]
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

# Build features
X_list, y_res = [], []
for rid in ALL_REGIONS:
    info = region_info[rid]; rt = REGION_TYPES[rid]
    for day_type in ['工作日','周末']:
        iw = 1 if day_type=='工作日' else 0
        for h in range(24):
            base = base_hourly.get((rid,h,day_type),0)
            actual = loads.get((rid,h,day_type),0)
            feats = [info['充电桩数量'], info['车流量'], info['人口密度'], info['商业POI数'], info['电网容量'],
                     np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24),
                     1 if 7<=h<=9 else 0, 1 if 17<=h<=20 else 0, iw,
                     1 if rt=='老城核心区' else 0, 1 if rt=='城市新区' else 0, 1 if rt=='城郊/工业区' else 0]
            X_list.append(feats); y_res.append(actual - base)
X, y = np.array(X_list), np.array(y_res)

# ====== Multi-seed evaluation ======
all_results = []
for seed in SEEDS:
    np.random.seed(seed)
    n = len(X); idx = np.random.permutation(n); split = int(n*0.8)
    Xtr, Xte = X[idx[:split]], X[idx[split:]]
    ytr, yte = y[idx[:split]], y[idx[split:]]
    model = xgb.XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5,
                             reg_lambda=1.0, random_state=seed, n_jobs=-1, verbosity=0)
    model.fit(Xtr, ytr)
    yp = model.predict(Xte)
    mae = mean_absolute_error(yte, yp)
    rmse = np.sqrt(mean_squared_error(yte, yp))
    r2 = r2_score(yte, yp)
    all_results.append({'seed': seed, 'MAE(kW)': mae, 'RMSE(kW)': rmse, 'R²': r2})
    print(f'  seed={seed:3d}: MAE={mae:.1f}kW, RMSE={rmse:.1f}kW, R²={r2:.4f}')

# Summary
mae_arr = np.array([r['MAE(kW)'] for r in all_results])
rmse_arr = np.array([r['RMSE(kW)'] for r in all_results])
r2_arr = np.array([r['R²'] for r in all_results])

summary = pd.DataFrame([
    {'指标': 'MAE(kW)', '均值': round(mae_arr.mean(),2), '标准差': round(mae_arr.std(),2),
     '最小值': round(mae_arr.min(),2), '最大值': round(mae_arr.max(),2)},
    {'指标': 'RMSE(kW)', '均值': round(rmse_arr.mean(),2), '标准差': round(rmse_arr.std(),2),
     '最小值': round(rmse_arr.min(),2), '最大值': round(rmse_arr.max(),2)},
    {'指标': 'R²', '均值': round(r2_arr.mean(),4), '标准差': round(r2_arr.std(),4),
     '最小值': round(r2_arr.min(),4), '最大值': round(r2_arr.max(),4)},
])

summary.to_excel(os.path.join(RESULTS_TABLES, 'robustness.xlsx'), index=False)
print(f'\n✅ 鲁棒性结果: {os.path.join(RESULTS_TABLES, "robustness.xlsx")}')
print(f'   MAE: {mae_arr.mean():.1f} ± {mae_arr.std():.1f} kW')
print(f'   RMSE: {rmse_arr.mean():.1f} ± {rmse_arr.std():.1f} kW')
print(f'   R²: {r2_arr.mean():.4f} ± {r2_arr.std():.4f}')
print('=' * 60)
print('鲁棒性测试完成！')
