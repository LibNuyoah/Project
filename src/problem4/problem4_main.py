"""
=============================================================================
problem4_main.py — 问题四：基于需求预测、布局优化与负荷调度协同的
                    充电网络生命周期动态扩展模型
=============================================================================
核心定位（与Q2区分）：
  Q2：初始建设"在哪里建"（一次最优布局）
  Q4：未来"何时扩建、扩多少"（动态扩展规划 + 滚动反馈）

Q3融合方式：
  读取Q3调度结果 → 有效容量模型 C_eff = C_raw / (1 - η×α)
  调度削峰 → 同等设施可承载更多需求 → 延缓扩容

健康度：
  4指标（需求满足/覆盖/安全/经济），熵权法客观赋权

动态反馈：
  每年扩容后更新容量，下一年基于新容量重算健康度
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys, warnings
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from utils.paths import (
    RESULTS_Q4, RESULTS_FIGURES, RESULTS_TABLES, FILE_PREDICTION_RESULT,
    FILE_Q4_FINAL_RESULT, FILE_Q4_DEMAND, FILE_Q4_HEALTH, FILE_Q4_EXPANSION,
    FILE_Q4_PRIORITY, FILE_Q4_CAPACITY, FILE_Q4_SCENARIO, FILE_Q4_PLAN
)

plt.rcParams.update({
    'font.sans-serif': ['Microsoft YaHei','SimHei','DejaVu Sans'],
    'axes.unicode_minus': False, 'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# 使用RESULTS_Q4作为输出目录
OUTPUT_DIR = RESULTS_Q4

# ========================================================================
# 常量与参数
# ========================================================================
REGION_NAMES = ['宝塔山街道','南市街道','凤凰山街道','桥沟街道','枣园街道',
                '新城街道','河庄坪镇','姚店镇（经开区）','万花山镇','真武洞街道（安塞）']
N = 10; YEARS = [2026, 2027, 2028]

# 增长率：基准15%（题目原文）±5%灵敏度
SCENARIOS = {'低增长(10%)':0.10, '基准增长(15%)':0.15, '高增长(20%)':0.20}
CS_COLOR = {'低增长(10%)':'#2B579A','基准增长(15%)':'#E07B39','高增长(20%)':'#C44E52'}

# 附件5参数
CAP_F=80; CAP_S=20; PW_F=120; PW_S=7; CST_F=6.0; CST_S=0.8
SIM=0.8; AVG_CHG=12.0; OVL=2100; COV_MIN=0.90

# Q3调度能力转化系数（可调参数）
ALPHA_DISPATCH = 0.65  # 调度削峰转化为有效容量提升的折扣因子

# 扩容触发阈值
TH_H = 0.72   # 健康度下限
TH_RHO = 0.85 # 服务压力上限

print('='*60)
print('问题四：充电网络生命周期动态扩展模型')
print('='*60)

# ========================================================================
# 1. 加载Q1-Q3数据
# ========================================================================
print('\n[1/6] 加载数据...')

# Q1预测
try:
    df_pred = pd.read_excel(FILE_PREDICTION_RESULT)
except:
    df_pred = pd.read_excel(FILE_PREDICTION_RESULT)
D0 = df_pred['预测日均需求_kWh'].values
P0 = df_pred['峰值负荷_kWh'].values

# Q2方案：从优化结果动态加载
try:
    df_q2 = pd.read_excel(os.path.join(RESULTS_TABLES, '表2_各区域最优配置方案.xlsx'))
    BF = df_q2['新增快充桩(台)'].values
    BS = df_q2['新增慢充桩(台)'].values
    COV = df_q2['地理覆盖率'].values
    EF = df_q2['现有快充桩(台)'].values
    ES = df_q2['现有慢充桩(台)'].values
    print('  Q2方案: 从表2动态加载')
except Exception:
    BF = np.array([0,0,2,0,0,2,0,0,1,1])
    BS = np.array([2,3,0,11,7,1,6,5,6,7])
    COV = np.array([0.9203,1.0,1.0,0.9193,0.9098,0.9466,1.0,0.9353,1.0,1.0])
    EF = np.array([129,119,99,109,76,95,45,59,39,53])
    ES = np.array([86,79,66,73,50,63,30,39,26,35])
    print('  Q2方案: 回退至默认值')

# 区域基础数据
AREA = np.array([17.36,14.25,17.62,110.07,80.10,60.08,139.87,120.04,131.20,22.30])
CINIT = np.array([14.02,11.10,14.50,55.03,32.44,41.89,35.02,42.00,26.17,14.50])
GRID = np.array([325000,298000,276000,352000,225000,308000,186000,255000,152000,205000])
RAD = np.array([1.5,1.5,2.0,1.5,2.0,2.0,2.5,2.5,2.5,2.5])
SCOV = np.pi*RAD**2

# Q3调度结果：工作日削峰31.2%、周末11.9%、综合≈20%
ETA_Q3 = 0.20
print(f'  Q3调度削峰率: {ETA_Q3*100:.0f}% | 有效容量转化系数α={ALPHA_DISPATCH}')

# ========================================================================
# 2. 多情景需求推演
# ========================================================================
print('\n[2/6] 多情景需求推演...')
recs = []
for sc, r in SCENARIOS.items():
    for t, yr in enumerate(YEARS):
        m = (1+r)**t
        for i in range(N):
            recs.append({'情景':sc,'年份':yr,'区域':i+1,'区域名称':REGION_NAMES[i],
                '日均需求_kWh':D0[i]*m,'峰值负荷_kW':P0[i]*m})
df_dem = pd.DataFrame(recs)
df_dem.to_excel(FILE_Q4_DEMAND, index=False)
print(f'  输出: {len(df_dem)}条')

# ========================================================================
# 3. 有效容量模型（Q3融合）+ 动态反馈循环
# ========================================================================
print('\n[3/6] 动态反馈循环：逐年评价→触发扩容→更新容量...')

# 熵权法函数
def entropy_weights(matrix):
    """4列矩阵 → 4维权重的熵权法"""
    n, m = matrix.shape
    w = np.ones(m) / m
    for j in range(m):
        col = matrix[:, j]
        mn, mx = col.min(), col.max()
        if mx - mn < 1e-8:
            w[j] = 0.0; continue
        p = (col - mn) / (mx - mn)
        p = np.clip(p, 1e-10, 1)
        e = -np.sum(p * np.log(p)) / np.log(n)
        w[j] = 1 - e
    return w / max(w.sum(), 1e-8)

# 初始化：每个情景维护独立的累计扩容数组
all_health = []
all_exp = []
all_priority = []
all_op_cap = []
final_summary = []

def dp_expand(H, rho, ov, trips, cap_eff, pk_d, cov_i, area_i, scov_i, cum_f, cum_s):
    """
    DP搜索最优扩容方案:
    min cost + 0.5×供需缺口 + 0.3×电网风险
    """
    best_action = (0, 0, 1e9)  # (nf, ns, min_cost)
    # 搜索空间: 快充0-20, 慢充0-40 (以5为步长控制搜索规模)
    for nf in range(0, 21, 5):
        for ns in range(0, 41, 5):
            if nf == 0 and ns == 0: continue
            # 计算扩容后的指标
            new_cap = cap_eff + CAP_F*nf + CAP_S*ns
            new_peak = pk_d + (PW_F*nf + PW_S*ns)*SIM
            # 成本
            cost = CST_F*nf + CST_S*ns
            # 供需缺口惩罚
            gap = max(0, trips - new_cap)
            # 电网风险惩罚
            risk = max(0, new_peak - OVL)
            # 覆盖提升
            cov_gain = (scov_i*(nf+ns)/area_i) if area_i>0 else 0
            cov_penalty = max(0, COV_MIN - (cov_i + cov_gain)) * 1000
            # 总目标
            total = cost + 0.5*gap + 0.3*risk + cov_penalty
            if total < best_action[2]:
                best_action = (nf, ns, total)
    return best_action[0], best_action[1]

for sc, r in SCENARIOS.items():
    cum_f = BF.copy().astype(float); cum_s = BS.copy().astype(float)
    for t, yr in enumerate(YEARS):
        m = (1+r)**t
        cap_raw = CAP_F*(EF + cum_f) + CAP_S*(ES + cum_s)
        cap_eff = cap_raw / max(1 - ETA_Q3 * ALPHA_DISPATCH, 0.01)
        cap_boost = (cap_eff - cap_raw) / np.maximum(cap_raw, 1) * 100
        trips = D0*m / AVG_CHG
        s1 = np.minimum(cap_eff / np.maximum(trips, 1), 1.0)
        s2 = COV.copy()
        pk = P0*m; pk_d = pk*(1-ETA_Q3)
        s3a = 1.0 - np.minimum(pk_d/GRID, 1.0)
        s3b = np.where(pk_d<=OVL, 1.0, np.maximum(0, 1.0-(pk_d-OVL)/OVL))
        s3 = np.minimum(s3a, s3b)
        invest_i = CST_F*cum_f + CST_S*cum_s
        s4 = 1.0 - invest_i / max(invest_i.max(), 1)
        mat = np.column_stack([s1, s2, s3, s4]); w_ent = entropy_weights(mat)
        H = w_ent[0]*s1 + w_ent[1]*s2 + w_ent[2]*s3 + w_ent[3]*s4
        rho = trips / np.maximum(cap_eff, 1); ov = (pk_d > OVL).astype(int)

        for i in range(N):
            pf = round(pk_d[i],1) if pk_d[i] <= OVL else OVL + 1
            all_op_cap.append({'情景':sc,'年份':yr,'区域':i+1,'区域名称':REGION_NAMES[i],
                '原始容量':round(cap_raw[i],0),'有效容量':round(cap_eff[i],0),
                '容量提升%':round(cap_boost[i],2),'调度效率η':ETA_Q3,'转化系数α':ALPHA_DISPATCH})
        for i in range(N):
            all_health.append({'情景':sc,'年份':yr,'区域':i+1,'区域名称':REGION_NAMES[i],
                'S1_需求满足':round(s1[i],4),'S2_覆盖':round(s2[i],4),'S3_安全':round(s3[i],4),
                'S4_经济':round(s4[i],4),'w1':round(w_ent[0],4),'w2':round(w_ent[1],4),
                'w3':round(w_ent[2],4),'w4':round(w_ent[3],4),'健康度_H':round(H[i],4),
                '服务压力':round(rho[i],4),'过载风险':ov[i],'峰调后_kW':round(pk_d[i],1),
                '有效容量':round(cap_eff[i],0)})
        for i in range(N):
            G_i = (D0[i]*(1+r)**2 - D0[i]) / D0[i]
            P_i = 0.5*(1-H[i]) + 0.3*rho[i] + 0.2*(G_i/0.40)
            all_priority.append({'情景':sc,'年份':yr,'区域':i+1,'区域名称':REGION_NAMES[i],
                '健康度':round(H[i],4),'饱和度':round(rho[i],4),'需求增速':f'{G_i*100:.1f}%',
                '优先级评分':round(P_i,4)})

        # DP扩容决策
        yr_exp = []
        for i in range(N):
            need = (H[i] < TH_H) or (rho[i] > TH_RHO) or (ov[i] == 1)
            if not need: continue
            nf, ns = dp_expand(H[i], rho[i], ov[i], trips[i], cap_eff[i],
                              pk_d[i], COV[i], AREA[i], SCOV[i], cum_f[i], cum_s[i])
            if nf == 0 and ns == 0: continue
            cum_f[i] += nf; cum_s[i] += ns
            yr_exp.append({'情景':sc,'年份':yr,'区域':i+1,'区域名称':REGION_NAMES[i],
                '健康度':round(H[i],4),'服务压力':round(rho[i],4),
                '新增快充':nf,'新增慢充':ns,'成本_万':CST_F*nf+CST_S*ns,
                '触发原因':'DP优化'})
            all_exp.append(yr_exp[-1])
        if yr_exp:
            df_tmp = pd.DataFrame(yr_exp)
            print(f'  {sc} {yr}年: 扩容{len(yr_exp)}区域, '
                  f'{df_tmp["新增快充"].sum()}快+{df_tmp["新增慢充"].sum()}慢, '
                  f'{df_tmp["成本_万"].sum():.1f}万')
    last_yr = 2028
    mh = [h for h in all_health if h['情景']==sc and h['年份']==last_yr]
    me = [e for e in all_exp if e['情景']==sc]
    avg_H = np.mean([h['健康度_H'] for h in mh])
    ov_cnt = sum(h['过载风险'] for h in mh)
    total_cost = sum(e['成本_万'] for e in me)
    dem_2028 = df_dem[(df_dem['情景']==sc)&(df_dem['年份']==2028)]['日均需求_kWh'].sum()/1000
    final_summary.append({'情景':sc,'增长率':f'{r*100:.0f}%',
        '2028需求_MWh':round(dem_2028,1),'2028平均健康度':round(avg_H,4),
        '扩容总次数':len(me),'扩容总成本_万':round(total_cost,1),'2028过载区域数':ov_cnt})

# 保存所有数据
df_health = pd.DataFrame(all_health)
df_health.to_excel(FILE_Q4_HEALTH, index=False)

df_exp = pd.DataFrame(all_exp)
df_exp.to_excel(FILE_Q4_EXPANSION, index=False)

df_pri = pd.DataFrame(all_priority)
df_pri = df_pri.sort_values('优先级评分', ascending=False)
df_pri.to_excel(FILE_Q4_PRIORITY, index=False)

df_op = pd.DataFrame(all_op_cap)
df_op.to_excel(FILE_Q4_CAPACITY, index=False)

df_sum = pd.DataFrame(final_summary)
df_sum.to_excel(FILE_Q4_SCENARIO, index=False)

# 未来三年扩展规划汇总表
plan_recs = []
for sc in SCENARIOS:
    r = SCENARIOS[sc]
    for yr in YEARS:
        for i in range(N):
            h_row = df_health[(df_health['情景']==sc)&(df_health['年份']==yr)&(df_health['区域']==i+1)]
            e_row = df_exp[(df_exp['情景']==sc)&(df_exp['年份']==yr)&(df_exp['区域']==i+1)]
            plan_recs.append({
                '情景':sc,'年份':yr,'区域':i+1,'区域名称':REGION_NAMES[i],
                '预测车次':round(D0[i]*(1+r)**(yr-2026)/AVG_CHG,0),
                '有效容量':round(h_row['有效容量'].values[0],0) if len(h_row)>0 else 0,
                '健康度':h_row['健康度_H'].values[0] if len(h_row)>0 else 0,
                '是否扩容':'是' if len(e_row)>0 else '否',
                '新增快充':int(e_row['新增快充'].sum()) if len(e_row)>0 else 0,
                '新增慢充':int(e_row['新增慢充'].sum()) if len(e_row)>0 else 0,
                '累计快充':int(EF[i]+BF[i]+sum(df_exp[(df_exp['情景']==sc)&(df_exp['年份']<=yr)&(df_exp['区域']==i+1)]['新增快充'])),
                '累计慢充':int(ES[i]+BS[i]+sum(df_exp[(df_exp['情景']==sc)&(df_exp['年份']<=yr)&(df_exp['区域']==i+1)]['新增慢充'])),
            })
pd.DataFrame(plan_recs).to_excel(FILE_Q4_PLAN, index=False)
print(f'  未来三年扩展规划: {len(plan_recs)}条')

print(f'\n  全部保存: 健康度{len(df_health)}条 | 扩容{len(df_exp)}条 | 优先级{len(df_pri)}条 | 有效容量{len(df_op)}条')

# ========================================================================
# 4. 综合输出文件
# ========================================================================
print('\n[4/6] 生成综合输出 problem4_final_result.xlsx...')
with pd.ExcelWriter(FILE_Q4_FINAL_RESULT) as writer:
    df_dem.to_excel(writer, sheet_name='未来需求', index=False)
    df_op.to_excel(writer, sheet_name='有效容量', index=False)
    df_health.to_excel(writer, sheet_name='健康度', index=False)
    df_pri.to_excel(writer, sheet_name='扩容优先级', index=False)
    df_exp.to_excel(writer, sheet_name='扩容方案', index=False)
    df_sum.to_excel(writer, sheet_name='情景汇总', index=False)
print('  problem4_final_result.xlsx (6 sheet)')

# ========================================================================
# 5. 可视化（5张图）
# ========================================================================
print('\n[5/6] 生成图表...')

# ---- 图1: 多情景需求增长曲线 ----
fig1,(a1,a2)=plt.subplots(1,2,figsize=(12,4.5))
for sc in SCENARIOS:
    td=[df_dem[(df_dem['情景']==sc)&(df_dem['年份']==y)]['日均需求_kWh'].sum()/1000 for y in YEARS]
    a1.plot(YEARS,td,'o-',color=CS_COLOR[sc],lw=2,ms=6,label=sc)
a1.set_xlabel('年份');a1.set_ylabel('全市日均需求(MWh)')
a1.set_title('(a)多情景全市充电需求增长',fontweight='bold');a1.legend(fontsize=8);a1.grid(alpha=.3,ls='--')
x=np.arange(N);w=.35
d26=df_dem[(df_dem['情景']=='基准增长(15%)')&(df_dem['年份']==2026)]['日均需求_kWh'].values/1000
d28=df_dem[(df_dem['情景']=='基准增长(15%)')&(df_dem['年份']==2028)]['日均需求_kWh'].values/1000
a2.bar(x-w/2,d26,w,label='2026',color='#8C8C8C');a2.bar(x+w/2,d28,w,label='2028(+32%)',color=CS_COLOR['基准增长(15%)'])
a2.set_xticks(x);a2.set_xticklabels([n[:3] for n in REGION_NAMES],fontsize=7)
a2.set_ylabel('日均需求(MWh)');a2.set_title('(b)基准增长各区域需求变化',fontweight='bold')
a2.legend(fontsize=8);a2.grid(axis='y',alpha=.3,ls='--')
plt.tight_layout();fig1.savefig(os.path.join(RESULTS_FIGURES, '图1_多情景需求增长曲线.png'));plt.close()
print('  图1: 多情景需求增长曲线')

# ---- 图2: 健康度热力图 ----
fig2,axs2=plt.subplots(1,3,figsize=(16,5))
for idx,sc in enumerate(SCENARIOS):
    hm=np.zeros((N,3))
    for ti,yr in enumerate(YEARS):
        hm[:,ti]=df_health[(df_health['情景']==sc)&(df_health['年份']==yr)]['健康度_H'].values
    sns.heatmap(hm,annot=True,fmt='.3f',cmap='RdYlGn',vmin=0.5,vmax=1.0,ax=axs2[idx],
        xticklabels=YEARS,yticklabels=[f'R{i+1}' for i in range(N)],
        linewidths=.5,linecolor='white',cbar_kws={'label':'健康度','shrink':0.8})
    axs2[idx].set_title(f'{sc}',fontweight='bold');axs2[idx].set_xlabel('年份')
plt.tight_layout();fig2.savefig(os.path.join(RESULTS_FIGURES, '图2_健康度热力图.png'));plt.close()
print('  图2: 健康度热力图')

# ---- 图3: 调度优化前后有效容量对比 ----
fig3,(a3a,a3b)=plt.subplots(1,2,figsize=(11,4.5))
cap_2026 = df_op[(df_op['情景']=='基准增长(15%)')&(df_op['年份']==2026)]
x3=np.arange(N);w3=.35
a3a.bar(x3-w3/2,cap_2026['原始容量']/1000,w3,label='原始容量',color='#8C8C8C')
a3a.bar(x3+w3/2,cap_2026['有效容量']/1000,w3,label='调度后有效容量',color='#2B579A')
a3a.set_xticks(x3);a3a.set_xticklabels([n[:3] for n in REGION_NAMES],fontsize=7)
a3a.set_ylabel('容量(千车次/日)');a3a.set_title('(a)调度前后各区域服务容量对比(2026)',fontweight='bold')
a3a.legend(fontsize=7);a3a.grid(axis='y',alpha=.3,ls='--')
# 容量提升比例
boost_2026=cap_2026.groupby('区域名称')['容量提升%'].mean().reindex(REGION_NAMES)
a3b.barh(range(N),boost_2026.values,color='#E07B39',edgecolor='white')
for i,v in enumerate(boost_2026.values):
    a3b.text(v+0.1,i,f'{v:.1f}%',va='center',fontsize=8)
a3b.set_yticks(range(N));a3b.set_yticklabels([n[:3] for n in REGION_NAMES],fontsize=7)
a3b.set_xlabel('容量提升(%)');a3b.set_title('(b)调度带来的容量提升比例',fontweight='bold');a3b.invert_yaxis()
plt.tight_layout();fig3.savefig(os.path.join(RESULTS_FIGURES, '图3_调度有效容量对比.png'));plt.close()
print('  图3: 调度有效容量对比')

# ---- 图4: 扩容优先级排序 ----
fig4,ax4=plt.subplots(figsize=(10,5.5))
bp=df_pri[(df_pri['情景']=='基准增长(15%)')&(df_pri['年份']==2028)].sort_values('优先级评分',ascending=True)
cols4=['#C44E52' if v>0.5 else '#E07B39' if v>0.3 else '#2B579A' for v in bp['优先级评分']]
ax4.barh(range(N),bp['优先级评分'].values,color=cols4,edgecolor='white')
for i,(v,n) in enumerate(zip(bp['优先级评分'],bp['区域名称'])):
    ax4.text(v+0.005,i,f'{n}({v:.3f})',va='center',fontsize=8,color='#333')
ax4.set_yticks(range(N));ax4.set_yticklabels([f'#{N-i}' for i in range(N)])
ax4.set_xlabel('优先级评分 P=0.5(1-H)+0.3ρ+0.2G',fontsize=9)
ax4.set_title('基准增长(15%)下各区域扩容优先级(2028年)',fontweight='bold')
ax4.invert_yaxis();ax4.set_xlim(0,0.9);ax4.grid(axis='x',alpha=.3,ls='--')
plt.tight_layout();fig4.savefig(os.path.join(RESULTS_FIGURES, '图4_扩容优先级排序.png'));plt.close()
print('  图4: 扩容优先级排序')

# ---- 图5: 三年动态扩容方案 ----
fig5,(a5a,a5b)=plt.subplots(1,2,figsize=(11,4.5))
for sc in SCENARIOS:
    m=df_exp['情景']==sc
    if not m.any():continue
    costs=[df_exp[m&(df_exp['年份']==y)]['成本_万'].sum() if (m&(df_exp['年份']==y)).any() else 0 for y in [2027,2028]]
    a5a.plot([2027,2028],costs,'o-',color=CS_COLOR[sc],lw=2,ms=8,label=sc)
a5a.set_xlabel('年份');a5a.set_ylabel('扩容成本(万元)')
a5a.set_title('(a)各情景累计扩容投资',fontweight='bold');a5a.legend(fontsize=8);a5a.grid(alpha=.3,ls='--')
# 基准增长：快充vs慢充堆叠
m5=df_exp['情景']=='基准增长(15%)'
if m5.any():
    yr_fast=[df_exp[m5&(df_exp['年份']==y)]['新增快充'].sum() for y in [2027,2028]]
    yr_slow=[df_exp[m5&(df_exp['年份']==y)]['新增慢充'].sum() for y in [2027,2028]]
    a5b.bar([2027,2028],yr_fast,label='快充',color='#2B579A')
    a5b.bar([2027,2028],yr_slow,bottom=yr_fast,label='慢充',color='#E07B39')
    for i,(y,f,s) in enumerate(zip([2027,2028],yr_fast,yr_slow)):
        a5b.text(y,f+s+0.5,f'{f+s}',ha='center',fontsize=9,fontweight='bold')
a5b.set_xlabel('年份');a5b.set_ylabel('新增桩数(台)')
a5b.set_title('(b)基准增长扩容构成(快充+慢充)',fontweight='bold');a5b.legend(fontsize=8)
plt.tight_layout();fig5.savefig(os.path.join(RESULTS_FIGURES, '图5_三年动态扩容方案.png'));plt.close()
print('  图5: 三年动态扩容方案')

# ========================================================================
# 6. 完成
# ========================================================================
print('\n[6/6] 完成！')
print('\n' + '='*60)
print('输出文件清单:')
for f in sorted(os.listdir(RESULTS_Q4)):
    kb = os.path.getsize(os.path.join(RESULTS_Q4, f))/1024
    print(f'  {f} ({kb:.1f}KB)')
print('='*60)
