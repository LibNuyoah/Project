# 城市新能源公共充电网络智能规划与调度

**2026年数学建模B题**

---

## 项目简介

本项目针对延安市主城区10个典型区域，研究城市新能源公共充电网络的智能规划与调度问题。

| 问题 | 内容 | 方法 | 状态 |
|:---|:---|:---|:---:|
| **问题1** | 充电需求分析与预测 | XGBoost + SHAP | ✅ |
| **问题2** | 充电桩多目标优化配置 | NSGA-II + 熵权-TOPSIS | ✅ |
| **问题3** | 分时电价调度与峰谷差优化 | 需求响应负荷转移 | ✅ |
| **问题4** | 充电网络生命周期动态扩展规划 | 多情景+健康度+动态反馈 | ✅ |

### 技术路线

```
原始附件(1-5)
     │
     ├── 问题1: 数据清洗 → EDA(空间/时间/工作日周末) → 聚类 → XGBoost → SHAP → 需求预测
     │        输出: 10区域日均充电需求
     │              ↓
     ├── 问题2: 供需缺口 → NSGA-II多目标优化 → TOPSIS决策 → 最优配置方案
     │        输出: 各区域新增快充/慢充数量
     │
     ├── 问题3: 峰谷差分析 → 20%负荷转移(均匀/填谷) → 效果评估
     │        输出: 调度前后负荷曲线、峰谷差降低率
     │
     └── 问题4: 多情景需求推演 → 有效容量修正(Q3融合) → 健康度评价 → 扩容触发 → 三年扩展方案
              输出: 5张图表 + 动态扩容方案
```

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 检查数据完整性
python utils/check_data.py

# 3. 运行全流程
python run_all.py

# 或按问题单独运行
python src/problem1/main.py                         # 问题1
python src/problem2/problem2_result.py              # 问题2
python src/problem3/problem3_result.py              # 问题3
python src/problem4/problem4_main.py                # 问题4
```

---

## 数据说明

### 数据来源

所有数据均来自官方附件，存放于 `data/raw/`，**未做任何拆分或改名**：

```
data/raw/
├── 附件 1 市主城区 10 个典型区域基础数据.xlsx          # 区域基础属性
├── 附件2 市主城区 10 区域分时段充电车次.xlsx           # 24h充电车次（2 sheet）
├── 附件3 市主城区 10 区域分时段充电负荷.xlsx           # 24h充电负荷（2 sheet）
├── 附件4 市主城区 10 个区域分时段电网最大允许负荷数据.xlsx  # 电网安全上限
└── 附件5  基础参数补充.docx                            # 成本/电价/覆盖率参数
```

### 附件2/3 读取方式

附件2和附件3各为一个Excel文件，包含**工作日**和**周末**两个sheet：

```python
from utils.paths import FILE_ATTACHMENT2, FILE_ATTACHMENT3

# 附件2：充电车次
df_wd = pd.read_excel(FILE_ATTACHMENT2, sheet_name='工作日分时段充电车次数据')
df_we = pd.read_excel(FILE_ATTACHMENT2, sheet_name='周末充电车次数据')

# 附件3：充电负荷
df_wd = pd.read_excel(FILE_ATTACHMENT3, sheet_name='工作日分时段充电负荷数据')
df_we = pd.read_excel(FILE_ATTACHMENT3, sheet_name='周末充电负荷数据（修改后）')
```

---

## 项目目录结构

```
project/
├── README.md                             # 项目说明
├── requirements.txt                      # Python依赖
├── run_all.py                            # 统一运行入口（全流程）
│
├── data/                                 # 数据目录
│   ├── raw/                              # 原始附件数据（5个文件，不可改名）
│   │   ├── 附件 1 ...xlsx               # 区域基础数据
│   │   ├── 附件2 ...xlsx                # 分时段充电车次
│   │   ├── 附件3 ...xlsx                # 分时段充电负荷
│   │   ├── 附件4 ...xlsx                # 电网最大允许负荷
│   │   └── 附件5 ...docx                # 基础参数补充
│   └── processed/                        # 数据预处理结果（程序生成）
│       └── clean_data.xlsx              # 清洗后的合并数据
│
├── models/                               # 模型文件
│   └── xgboost_model.pkl                # 训练好的XGBoost模型
│
├── results/                              # 输出结果目录
│   ├── figures/                          # 图片（.png）
│   ├── tables/                           # 表格（.xlsx）
│   ├── logs/                             # 日志
│   ├── q3_output/                        # 问题3中间数据与图表
│   ├── q4_output/                        # 问题4输出图表
│   └── prediction_result.xlsx           # 问题1最终预测结果
│
├── src/                                  # 源代码
│   ├── problem1/                         # 问题1：充电需求分析与预测
│   │   ├── main.py                       # Q1主入口（预测汇总输出）
│   │   ├── preprocess/
│   │   │   └── data_clean.py             # 数据读取、清洗、格式转换
│   │   ├── analysis/
│   │   │   ├── spatial_analysis.py       # 空间维度分析 + 单桩利用率
│   │   │   ├── temporal_analysis.py      # 时间维度 + 工作日/周末差异
│   │   │   ├── correlation_analysis.py   # Pearson相关 + 热力图
│   │   │   ├── cluster_analysis.py       # K-means区域功能聚类
│   │   │   └── region_type_loader.py     # 聚类结果加载器
│   │   └── model/
│   │       ├── xgboost_model.py          # XGBoost预测（含超参调优）
│   │       └── shap_analysis.py          # SHAP模型解释
│   │
│   ├── problem2/                         # 问题2：充电桩优化配置
│   │   ├── problem2_data.py              # 供需缺口 + 空间溢出矩阵
│   │   ├── problem2_optimize.py          # NSGA-II多目标优化求解
│   │   └── problem2_result.py            # TOPSIS决策 + 可视化
│   │
│   ├── problem3/                         # 问题3：分时电价调度
│   │   ├── problem3_data.py              # 调度前峰谷差分析
│   │   ├── problem3_solve.py             # 负荷转移（均匀/填谷两方案）
│   │   └── problem3_result.py            # 效果评估 + 可视化
│   │
│   └── problem4/                         # 问题4：生命周期动态扩展规划
│       └── problem4_main.py              # 多情景推演 + 健康度 + 扩容
│
├── utils/                                # 公共工具
│   ├── __init__.py                       # 包初始化
│   ├── paths.py                          # 统一路径管理（所有代码引用路径的唯一入口）
│   └── check_data.py                     # 数据完整性检查
│
└── docs/                                 # 论文与参考资料
    ├── 2026_B题.docx                     # 赛题原文
    ├── B题论文_v10.docx                  # 最新论文
    ├── B题论文_v6~v9.docx                # 历史论文版本
    ├── problem2框架.md                   # 问题2模型框架
    ├── problem2最终论文.docx             # 问题2定稿
    ├── 问题3框架.md                      # 问题3模型框架
    ├── run_all修复说明.md                # 运行修复记录
    └── 工作日志.md                        # 小组工作日志
```

---

## 运行顺序

各问题之间存在数据依赖关系，必须按顺序运行：

```
1. utils/check_data.py                      # 数据完整性检查（必须先运行）
       ↓
2. src/problem1/preprocess/data_clean.py    # 数据清洗
3. src/problem1/analysis/cluster_analysis.py  # 区域聚类
4. src/problem1/model/xgboost_model.py      # XGBoost训练
5. src/problem1/model/shap_analysis.py      # SHAP解释
6. src/problem1/main.py                     # 预测汇总输出
       ↓
7. src/problem2/problem2_data.py            # 供需分析（依赖Q1预测结果）
8. src/problem2/problem2_optimize.py        # NSGA-II优化
9. src/problem2/problem2_result.py          # 结果可视化
       ↓
10. src/problem3/problem3_data.py           # 调度前分析（依赖附件3）
11. src/problem3/problem3_solve.py          # 负荷转移
12. src/problem3/problem3_result.py         # 效果评估
       ↓
13. src/problem4/problem4_main.py            # 动态扩展规划（依赖Q1-Q3）
```

或直接运行 `python run_all.py` 一键完成全部流程。

---

## 问题1：充电需求估计模型

### 模型架构

| 项目 | 说明 |
|:---|:---|
| **主模型** | 80/20 随机划分（random_state=42），XGBoost |
| **补充实验** | GroupKFold 区域留一交叉验证（GridSearch 21870 fits） |
| **目标变换** | log1p（压缩充电负荷长尾分布） |
| **特征维度** | 28维（空间14 + 时间10 + 先验1 + 聚类3） |

### 特征体系

| 类别 | 维度 | 内容 |
|:---|:---:|:---|
| 空间基础 | 9 | 人口密度、车流量、POI、充电桩、快慢充、电网、面积 |
| 空间派生 | 5 | 充电桩密度、快充比例、交通强度、商业密度、设施供给强度 |
| 时间周期 | 10 | sin/cos周期编码 + 早晚高峰 + 时间段OneHot(5) + 是否工作日 |
| 区域先验 | 1 | 区域历史平均负荷（log1p尺度） |
| 区域功能 | 3 | KMeans聚类One-Hot |

### 主模型性能（80/20）

| 指标 | 值 |
|:---|---:|
| 测试 R² | 0.9333 |
| 测试 MAE | ~82 kWh |
| 测试 RMSE | 120.7 kWh |
| 测试 SMAPE | 工作日 22.0% / 周末 38.9% |
| 测试 RPD | 3.87（优） |

### 最优参数

```python
{
    'colsample_bytree': 0.7, 'learning_rate': 0.03,
    'max_depth': 5, 'n_estimators': 300,
    'reg_alpha': 0.5, 'reg_lambda': 1, 'subsample': 0.7
}
```

---

## 路径管理

所有Python文件通过 `utils/paths.py` 统一管理项目路径：

```python
from utils.paths import (
    DATA_RAW,           # 原始数据目录
    DATA_PROCESSED,     # 处理后的数据目录
    RESULTS_DIR,        # 结果根目录
    RESULTS_FIGURES,    # 图片目录
    RESULTS_TABLES,     # 表格目录
    FILE_ATTACHMENT1,   # 附件1路径
    FILE_CLEAN_DATA,    # 清洗数据路径
    FILE_PREDICTION_RESULT,  # 预测结果路径
    ...
)
```

**禁止在任何代码中硬编码文件路径。** 所有路径必须通过 `utils/paths` 获取。

---

## 环境依赖

- Python 3.9+
- pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, shap, scipy, openpyxl
- 详见 `requirements.txt`
