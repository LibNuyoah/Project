# 城市新能源公共充电网络智能规划与调度

**2026年数学建模B题**

---

## 项目简介

本项目针对延安市主城区10个典型区域，研究城市新能源公共充电网络的智能规划与调度问题。

| 问题 | 内容 | 核心方法 | 状态 |
|:---|:---|:---|:---:|
| **问题1** | 充电需求分析与预测 | 双层模型：LOO物理基准 + XGBoost残差拟合 | ✅ |
| **问题2** | 充电桩多目标优化配置 | NSGA-II(4目标) + 熵权-TOPSIS | ✅ |
| **问题3** | 分时电价调度与峰谷差优化 | 均匀分配 + 经济调度(价格弹性) | ✅ |
| **问题4** | 生命周期动态扩展规划 | DP动态规划 + 多情景推演 + 健康度评价 | ✅ |

### 10个区域

| 编号 | 区域名称 | 类型 |
|:---:|------|------|
| 1 | 宝塔山街道 | 老城核心区 |
| 2 | 南市街道 | 老城核心区 |
| 3 | 凤凰山街道 | 老城核心区 |
| 4 | 桥沟街道 | 城市新区 |
| 5 | 枣园街道 | 城市新区 |
| 6 | 新城街道 | 城市新区 |
| 7 | 河庄坪镇 | 城郊/工业区 |
| 8 | 姚店镇（经开区） | 城郊/工业区 |
| 9 | 万花山镇 | 城郊/工业区 |
| 10 | 真武洞街道（安塞） | 城郊/工业区 |

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 一键运行全流程
python run_all.py

# 或按问题运行
python src/problem1/model/two_layer_model.py      # 问题1
python src/problem2/problem2_result.py             # 问题2
python src/problem3/problem3_result.py             # 问题3
python src/problem4/problem4_main.py               # 问题4

# 消融实验、鲁棒性验证与敏感性分析
python src/analysis/ablation.py
python src/analysis/robustness_test.py
python src/analysis/sensitivity.py
```

---

## 技术架构

```
原始附件(1-4)
     │
     ├── 问题1: LOO留一法日总量预测 → 同类区域小时分配 → XGBoost残差修正
     │        输出: 10区域日均充电需求 (±2.1%偏差)
     │              ↓
     ├── 问题2: 供需缺口分析 → NSGA-II(4目标:成本+覆盖+均衡+电网风险) → TOPSIS最优解
     │        输出: 各区域新增快充/慢充数量 (覆盖率98%, 过载不恶化)
     │
     ├── 问题3: 峰谷差分析 → 方案A(20%均匀) vs 方案B(经济调度)
     │        输出: 工作日削峰31.2%, 负荷率51.2%
     │
     └── 问题4: 多情景(10%/15%/20%)需求推演 → DP动态扩容 → 健康度评价
              输出: 3年扩展方案 + 5张图表
```

---

## 项目结构

```
project/
├── README.md
├── requirements.txt
├── run_all.py                          # 全流程一键运行
│
├── data/
│   ├── raw/                            # 原始附件（不可改名）
│   │   ├── 附件 1 市主城区 10 个典型区域基础数据.xlsx
│   │   ├── 附件2 市主城区 10 区域分时段充电车次.xlsx
│   │   ├── 附件3 市主城区 10 区域分时段充电负荷.xlsx
│   │   └── 附件4 市主城区 10 个区域分时段电网最大允许负荷数据.xlsx
│   └── processed/
│       └── clean_data.xlsx
│
├── models/
│   └── xgboost_model.pkl               # XGBoost残差模型
│
├── results/
│   ├── prediction_result.xlsx          # Q1: 10区域预测汇总表
│   ├── figures/                        # Q1+Q2+分析: 15张图表
│   ├── tables/                         # Q1+Q2+分析: 13张数据表
│   ├── q3_output/                      # Q3: 5张图 + 3张表
│   └── q4_output/                      # Q4: 5张图 + 7张表
│
├── src/
│   ├── problem1/
│   │   ├── preprocess/data_clean.py    # 数据清洗
│   │   ├── analysis/                   # EDA分析
│   │   │   ├── cluster_analysis.py     # 区域聚类
│   │   │   └── region_type_loader.py   # 区域类型加载
│   │   └── model/
│   │       └── two_layer_model.py      # ★ 双层预测模型（主入口）
│   │
│   ├── problem2/
│   │   ├── problem2_data.py            # 供需缺口 + 溢出矩阵
│   │   ├── problem2_optimize.py        # NSGA-II 4目标优化
│   │   └── problem2_result.py          # TOPSIS + 可视化
│   │
│   ├── problem3/
│   │   ├── problem3_data.py            # 调度前分析
│   │   ├── problem3_solve.py           # A(均匀) vs B(经济调度)
│   │   └── problem3_result.py          # 效果评估
│   │
│   ├── problem4/
│   │   └── problem4_main.py            # DP动态扩展规划
│   │
│   └── analysis/
│       ├── ablation.py                 # 模型消融实验
│       ├── robustness_test.py          # 鲁棒性验证
│       └── sensitivity.py              # 敏感性分析
│
├── utils/
│   ├── paths.py                        # 统一路径管理
│   └── check_data.py                   # 数据完整性检查
│
└── docs/                               # 论文与参考资料
```

---

## 问题1：双层充电需求预测模型

### 架构

```
最终预测 = Layer1(物理基准) + Layer2(XGBoost残差)
```

| 层 | 方法 | 参数量 | 作用 |
|---|------|:---:|------|
| Layer 1 | LOO留一法Ridge回归 + 同类区域能耗模式 | 3 | 量级受控、曲线正确 |
| Layer 2 | XGBoost残差拟合 (13维特征) | 150 | 修正局部偏差 |

### 为什么双层？

- **纯ML问题**：XGBoost直接预测kWh → 量级失控(25000 vs 7000)、凌晨虚假峰值
- **纯物理模型**：LOO日总量准确但小时分配粗糙
- **双层模型**：物理打底保证量级 + ML修正残差(仅±400kW) → 偏差<3%

### 性能

| 指标 | 值 |
|:---|---:|
| 预测偏差 | **-2.1% ~ +2.0%** |
| 全市总偏差 | **-0.19%** |
| XGBoost残差MAE | 60.4 kW |
| 残差R² | 0.527 |

### 鲁棒性（5种子验证）

| 指标 | 均值 ± 标准差 |
|:---|---:|
| MAE | 60.2 ± 1.7 kW |
| RMSE | 82.9 ± 4.0 kW |
| R² | 0.644 ± 0.067 |

### 输出成果

| 类型 | 文件 | 说明 |
|:---|------|------|
| 📊 图 | `results/figures/prediction_summary.png` | 四面板汇总图：日均需求柱状图 + 24h曲线 + 类型占比 + 报告 |
| 📊 图 | `results/figures/xgboost_evaluation.png` | XGBoost残差拟合评价：散点图 + 特征重要性 + 残差分布 + 参数 |
| 📊 图 | `results/figures/cluster_analysis.png` | 区域聚类分析：肘部法则 + PCA + 树状图 + 特征画像 + 轮廓系数 |
| 📋 表 | `results/prediction_result.xlsx` | 10区域预测汇总（含附件3真实值对比列） |
| 📋 表 | `results/tables/hourly_prediction.xlsx` | 480条分时段预测（10区×24h×2日期类型） |
| 📋 表 | `results/tables/xgboost_metrics.xlsx` | XGBoost训练指标（MAE/RMSE/R²） |
| 📋 表 | `results/tables/cluster_result.xlsx` | 区域功能聚类结果 |
| 💾 模型 | `models/xgboost_model.pkl` | 训练好的XGBoost残差预测模型 |

---

## 问题2：NSGA-II 四目标优化

### 目标函数

| # | 目标 | 方向 | 说明 |
|:--:|------|:--:|------|
| f1 | 建设成本 | ↓ | 快充6万/台, 慢充0.8万/台 |
| f2 | 地理覆盖率 | ↑ | 各区域独立覆盖率的均值 |
| f3 | 负荷均衡 | ↓ | 负载率方差 + 最大负载率惩罚 |
| f4 | 电网风险 | ↓ | Σ max(0, 负载率-90%)² |

### 约束

- 服务能力 ≥ 预测需求
- 配电过载：新增后峰值 ≤ 2100kW（已超区域禁止加快充）
- 覆盖率作为优化目标（非硬约束），避免与过载约束冲突

### 结果

| 指标 | 优化前 | 优化后 |
|------|:------:|:------:|
| 平均覆盖率 | 54.6% | **98.1%** |
| 过载风险区域 | 2 | **2** (未恶化) |
| 总投资 | — | 213.2万元 |

### 输出成果

| 类型 | 文件 | 说明 |
|:---|------|------|
| 📊 图 | `results/figures/图1_建设紧迫度与供需缺口.png` | 双面板：紧迫度排序 + 供需缺口柱状图 |
| 📊 图 | `results/figures/图2_空间溢出权重热力图.png` | 10×10空间溢出权重矩阵热力图 |
| 📊 图 | `results/figures/图3_NSGA-II求解过程.png` | 三面板：收敛曲线 + Pareto前沿 + 平行坐标 |
| 📊 图 | `results/figures/图4_TOPSIS最优解选取.png` | 双面板：TOPSIS评分 + 熵权法权重饼图 |
| 📊 图 | `results/figures/图5_配置方案与优化效果对比.png` | 双面板：配置方案对比 + 优化前后指标 |
| 📋 表 | `results/tables/表1_各区域供需缺口与建设紧迫度.xlsx` | 供需缺口/电网容量/紧迫度排名 |
| 📋 表 | `results/tables/表2_各区域最优配置方案.xlsx` | TOPSIS最优解：各区域新增快充/慢充数 |
| 📋 表 | `results/tables/表3_优化前后多指标对比.xlsx` | 覆盖率/过载/负荷率等6项指标对比 |
| 📋 表 | `results/tables/Pareto前沿解集.xlsx` | 100个Pareto非支配解完整数据 |
| 📋 表 | `results/tables/NSGA-II收敛曲线数据.xlsx` | 500代收敛过程记录 |
| 📋 表 | `results/tables/空间溢出权重矩阵.xlsx` | 基于距离矩阵的10×10溢出权重 |
| 📋 表 | `results/tables/区域距离矩阵.xlsx` | 10区域间欧氏距离矩阵 |

---

## 问题3：分时电价调度

### 两方案对比（工作日）

| 方案 | 方法 | 峰谷差降低 | 负荷率 |
|------|------|:---:|:---:|
| A 均匀迁移 | 20%高峰负荷均匀分配到低谷 | **31.2%** | 51.2% |
| B 经济调度 | 价格信号引导(谷0.5/平1.0/峰1.5) | 30.0% | 58.5% |

### 输出成果

| 类型 | 文件 | 说明 |
|:---|------|------|
| 📊 图 | `results/q3_output/图15_全市调度前后负荷曲线对比.png` | 调度前后全市24h负荷曲线对比 |
| 📊 图 | `results/q3_output/图16_各区域峰谷差降低率.png` | 10区域峰谷差降低率柱状图 |
| 📊 图 | `results/q3_output/图17_各区域分面负荷曲线.png` | 10区域×2日期类型分面负荷曲线 |
| 📊 图 | `results/q3_output/图18_各区域峰谷差对比.png` | 调度前后各区域峰谷差对比 |
| 📊 图 | `results/q3_output/图19_过载风险消除效果.png` | 过载风险消除效果展示 |
| 📋 表 | `results/q3_output/表_调度前分析.xlsx` | 调度前峰谷差/负荷率/过载统计 |
| 📋 表 | `results/q3_output/表A_调度前后峰谷差对比.xlsx` | 方案A/B调度前后峰谷差对比 |
| 📋 表 | `results/q3_output/表B_过载风险评估.xlsx` | 各区域过载风险评估结果 |

---

## 问题4：DP动态扩展规划

### 方法

- **状态**: 当前容量, 需求增长率, 健康度, 电网压力
- **动作**: (新增快充数, 新增慢充数)
- **目标**: min(建设成本 + 0.5×供需缺口 + 0.3×电网风险)
- **搜索**: 快充 0-20 × 慢充 0-40 网格搜索

### 情景

| 情景 | 增长率 | 2027扩容 | 2028扩容 | 总成本 |
|------|:---:|:---:|:---:|:---:|
| 低增长 | 10% | 0区域 | 1区域 | 13.6万 |
| 基准 | 15% | 1区域 | 2区域 | 43.6万 |
| 高增长 | 20% | 1区域 | 4区域 | 79.6万 |

### 输出成果

| 类型 | 文件 | 说明 |
|:---|------|------|
| 📊 图 | `results/q4_output/图1_多情景需求增长曲线.png` | 三情景全市需求增长 + 各区域2026vs2028对比 |
| 📊 图 | `results/q4_output/图2_健康度热力图.png` | 三情景×3年 10区域健康度热力图 |
| 📊 图 | `results/q4_output/图3_调度有效容量对比.png` | 调度前后服务容量对比 + 容量提升比例 |
| 📊 图 | `results/q4_output/图4_扩容优先级排序.png` | 基准增长下2028年各区域扩容优先级 |
| 📊 图 | `results/q4_output/图5_三年动态扩容方案.png` | 各情景投资曲线 + 快充/慢充堆叠构成 |
| 📋 表 | `results/q4_output/problem4_final_result.xlsx` | 综合输出（6 sheet：需求/容量/健康度/优先级/方案/汇总） |
| 📋 表 | `results/q4_output/未来需求推演.xlsx` | 3情景×3年×10区域需求推演（90条） |
| 📋 表 | `results/q4_output/健康度评价.xlsx` | 四维健康度指标详细数据 |
| 📋 表 | `results/q4_output/有效容量分析.xlsx` | 调度前后有效容量对比 |
| 📋 表 | `results/q4_output/扩容优先级.xlsx` | 各区域扩容优先级评分 |
| 📋 表 | `results/q4_output/动态扩容方案.xlsx` | DP优化后的逐年扩容方案 |
| 📋 表 | `results/q4_output/未来三年扩展规划.xlsx` | 完整三年扩展规划汇总 |
| 📋 表 | `results/q4_output/情景对比汇总.xlsx` | 三种情景关键指标对比 |

---

## 消融实验

| 模型变体 | MAE(kW) | RMSE(kW) | R² |
|------|:--:|:--:|:--:|
| Baseline XGBoost | 71.8 | 105.3 | 0.942 |
| 无时间特征 | 81.1 | 107.4 | 0.249 |
| 无空间特征 | 90.3 | 135.5 | -0.196 |
| 单层XGBoost | 71.8 | 105.3 | 0.942 |
| **双层模型(完整)** | **60.4** | **85.2** | **0.527** |

> 注：Baseline和单层XGBoost的R²是对原始负荷直接拟合的指标（目标尺度大），双层模型的R²是对残差的拟合指标（目标仅±400kW），不可直接对比R²。应以MAE/RMSE为准。

---

## 敏感性分析

分析关键参数对模型输出的影响，评估结论的稳健性。

### Q1: XGBoost超参数敏感性

| 参数 | 测试范围 | 最优值 | MAE影响 |
|------|:---:|:---:|:---:|
| n_estimators | 50-300 | **300** | 62→58 kW (-6%) |
| max_depth | 3-8 | **8** | 63→58 kW (-8%) |
| learning_rate | 0.01-0.10 | **0.10** | 65→58 kW (-11%) |

> 当前采用保守参数(n=150, d=4, lr=0.05)以平衡精度与泛化。
> 若追求极致精度可用(300, 8, 0.10)，MAE从60.4降至~58kW，但过拟合风险增加。

### Q2: 充电桩成本比敏感性

| 快充成本(万) | 快/慢成本比 | 推荐快充占比 |
|:---:|:---:|:---:|
| 3 | 3.8x | 47% |
| 6 | 7.5x | 31% |
| 9 | 11.3x | 23% |
| 12 | 15.0x | 18% |

> 快充成本越低，推荐占比越高。当前6万/0.8万(7.5x)对应约31%快充比例，与NSGA-II优化结果一致。

### Q3: 调度转移率敏感性

| 峰谷电价比 | 最优转移率 | 峰谷差降低率 |
|:---:|:---:|:---:|
| 2x | 30% | 30% |
| 3x | 35% | 35% |
| 5x | **40%** | **40%** |

> 电价比越高(峰时越贵)，最优转移率越高。当前采用20%转移率偏保守，若电价差扩大至5x，可提升至40%转移率。

### 输出成果

| 类型 | 文件 | 说明 |
|:---|------|------|
| 📊 图 | `results/figures/ablation.png` | 消融实验三面板柱状图（MAE/RMSE/R²对比） |
| 📊 图 | `results/figures/sensitivity_analysis.png` | 敏感性分析三面板（超参数/成本比/转移率） |
| 📋 表 | `results/tables/模型消融实验.xlsx` | 5种模型变体完整对比数据 |
| 📋 表 | `results/tables/robustness.xlsx` | 5种子鲁棒性验证统计（均值/标准差/最大/最小） |
| 📋 表 | `results/tables/敏感性分析.xlsx` | 三sheet敏感性数据（Q1超参/Q2成本比/Q3转移率） |

---

## 运行顺序与依赖

```
1. utils/check_data.py                       # 数据完整性检查
2. src/problem1/preprocess/data_clean.py     # 数据清洗
3. src/problem1/analysis/cluster_analysis.py # 区域聚类
4. src/problem1/model/two_layer_model.py     # ★ 双层预测（覆盖旧xgboost+main）
       ↓ 输出: prediction_result.xlsx
5. src/problem2/problem2_data.py             # 供需分析
6. src/problem2/problem2_optimize.py         # NSGA-II优化
7. src/problem2/problem2_result.py           # TOPSIS+可视化
       ↓
8. src/problem3/problem3_data.py             # 调度前分析
9. src/problem3/problem3_solve.py            # 负荷转移
10. src/problem3/problem3_result.py           # 效果评估
       ↓
11. src/problem4/problem4_main.py             # DP动态扩展

# 辅助分析（可选）
12. src/analysis/ablation.py                  # 模型消融实验
13. src/analysis/robustness_test.py           # 鲁棒性验证
14. src/analysis/sensitivity.py               # 敏感性分析
```

---

## 数据说明

### 附件读取方式

附件2和附件3各含两个sheet（工作日/周末），通过 `sheet_name` 参数区分：

```python
from utils.paths import FILE_ATTACHMENT2, FILE_ATTACHMENT3

# 附件2：充电车次
df_wd = pd.read_excel(FILE_ATTACHMENT2, sheet_name='工作日分时段充电车次数据')
df_we = pd.read_excel(FILE_ATTACHMENT2, sheet_name='周末充电车次数据')

# 附件3：充电负荷
df_wd = pd.read_excel(FILE_ATTACHMENT3, sheet_name='工作日分时段充电负荷数据')
df_we = pd.read_excel(FILE_ATTACHMENT3, sheet_name='周末充电负荷数据（修改后）')
```

### 路径管理

所有路径通过 `utils/paths.py` 统一管理，禁止硬编码：

```python
from utils.paths import (
    FILE_PREDICTION_RESULT,  # results/prediction_result.xlsx
    FILE_HOURLY_PREDICTION,  # results/tables/hourly_prediction.xlsx
    FILE_XGBOOST_MODEL,      # models/xgboost_model.pkl
    ...
)
```

---

## 环境依赖

- Python 3.9+
- pandas, numpy, scikit-learn, xgboost, matplotlib, scipy, openpyxl
- 详见 `requirements.txt`
