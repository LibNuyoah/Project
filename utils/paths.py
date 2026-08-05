"""
项目路径管理模块
---------------
统一管理所有项目路径。从任意位置 import 均能正确定位。

用法:
    from utils.paths import DATA_RAW, FILE_ATTACHMENT1, RESULTS_DIR
"""

import os

# ═══════════════════════════════════════════════════════════════
# 项目根目录
# ═══════════════════════════════════════════════════════════════
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ═══════════════════════════════════════════════════════════════
# 数据目录
# ═══════════════════════════════════════════════════════════════
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
DATA_RAW = os.path.join(DATA_DIR, 'raw')              # 原始附件数据
DATA_PROCESSED = os.path.join(DATA_DIR, 'processed')  # 数据预处理结果

# ═══════════════════════════════════════════════════════════════
# 模型目录（存放训练好的 .pkl 等二进制文件）
# ═══════════════════════════════════════════════════════════════
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# ═══════════════════════════════════════════════════════════════
# 输出目录
# ═══════════════════════════════════════════════════════════════
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
RESULTS_FIGURES = os.path.join(RESULTS_DIR, 'figures')     # 图片
RESULTS_TABLES = os.path.join(RESULTS_DIR, 'tables')        # 表格
RESULTS_LOGS = os.path.join(RESULTS_DIR, 'logs')            # 日志
RESULTS_Q3 = os.path.join(RESULTS_DIR, 'q3_output')         # 问题3中间数据
RESULTS_Q4 = os.path.join(RESULTS_DIR, 'q4_output')         # 问题4输出

# ═══════════════════════════════════════════════════════════════
# 文档目录
# ═══════════════════════════════════════════════════════════════
DOCS_DIR = os.path.join(PROJECT_ROOT, 'docs')

# ═══════════════════════════════════════════════════════════════
# 各问题源码目录
# ═══════════════════════════════════════════════════════════════
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
SRC_PROBLEM1 = os.path.join(SRC_DIR, 'problem1')
SRC_PROBLEM2 = os.path.join(SRC_DIR, 'problem2')
SRC_PROBLEM3 = os.path.join(SRC_DIR, 'problem3')
SRC_PROBLEM4 = os.path.join(SRC_DIR, 'problem4')

# ═══════════════════════════════════════════════════════════════
# 官方原始附件路径（不可改名、不可拆分）
# 附件2和附件3各含两个sheet，通过 sheet_name 参数区分 工作日/周末
# ═══════════════════════════════════════════════════════════════

# 附件1：区域基础数据
FILE_ATTACHMENT1 = os.path.join(
    DATA_RAW, '附件 1 市主城区 10 个典型区域基础数据.xlsx'
)
# 附件2：分时段充电车次（含工作日/周末两个sheet）
FILE_ATTACHMENT2 = os.path.join(
    DATA_RAW, '附件2 市主城区 10 区域分时段充电车次.xlsx'
)
# 附件3：分时段充电负荷（含工作日/周末两个sheet）
FILE_ATTACHMENT3 = os.path.join(
    DATA_RAW, '附件3 市主城区 10 区域分时段充电负荷.xlsx'
)
# 附件4：电网最大允许负荷
FILE_ATTACHMENT4 = os.path.join(
    DATA_RAW, '附件4 市主城区 10 个区域分时段电网最大允许负荷数据.xlsx'
)
# 附件5：基础参数补充
FILE_ATTACHMENT5 = os.path.join(
    DATA_RAW, '附件5  基础参数补充.docx'
)

# ═══════════════════════════════════════════════════════════════
# 附件2 Sheet 名称
# ═══════════════════════════════════════════════════════════════
SHEET_ATTACHMENT2_WEEKDAY = '工作日分时段充电车次数据'
SHEET_ATTACHMENT2_WEEKEND = '周末充电车次数据'

# ═══════════════════════════════════════════════════════════════
# 附件3 Sheet 名称
# ═══════════════════════════════════════════════════════════════
SHEET_ATTACHMENT3_WEEKDAY = '工作日分时段充电负荷数据'
SHEET_ATTACHMENT3_WEEKEND = '周末充电负荷数据（修改后）'

# ═══════════════════════════════════════════════════════════════
# 处理后数据文件
# ═══════════════════════════════════════════════════════════════
FILE_CLEAN_DATA = os.path.join(DATA_PROCESSED, 'clean_data.xlsx')

# ═══════════════════════════════════════════════════════════════
# 模型文件
# ═══════════════════════════════════════════════════════════════
FILE_XGBOOST_MODEL = os.path.join(MODELS_DIR, 'xgboost_model.pkl')

# ═══════════════════════════════════════════════════════════════
# 结果文件
# ═══════════════════════════════════════════════════════════════
FILE_PREDICTION_RESULT = os.path.join(RESULTS_DIR, 'prediction_result.xlsx')
FILE_CLUSTER_RESULT = os.path.join(RESULTS_TABLES, 'cluster_result.xlsx')
FILE_CORRELATION_MATRIX = os.path.join(RESULTS_TABLES, 'correlation_matrix.xlsx')
FILE_XGBOOST_METRICS = os.path.join(RESULTS_TABLES, 'xgboost_metrics.xlsx')
FILE_SHAP_IMPORTANCE = os.path.join(RESULTS_TABLES, 'shap_importance.xlsx')
FILE_HOURLY_PREDICTION = os.path.join(RESULTS_TABLES, 'hourly_prediction.xlsx')
FILE_MODEL_COMPARISON = os.path.join(RESULTS_TABLES, 'model_comparison.xlsx')
FILE_STANDARDIZATION_PARAMS = os.path.join(RESULTS_TABLES, 'standardization_params.xlsx')

# 问题2结果文件
FILE_Q2_TABLE1 = os.path.join(RESULTS_TABLES, '表1_各区域供需缺口与建设紧迫度.xlsx')
FILE_Q2_TABLE2 = os.path.join(RESULTS_TABLES, '表2_各区域最优配置方案.xlsx')
FILE_Q2_TABLE3 = os.path.join(RESULTS_TABLES, '表3_优化前后多指标对比.xlsx')
FILE_Q2_PARETO = os.path.join(RESULTS_TABLES, 'Pareto前沿解集.xlsx')
FILE_Q2_CONVERGENCE = os.path.join(RESULTS_TABLES, 'NSGA-II收敛曲线数据.xlsx')
FILE_Q2_SPILLOVER = os.path.join(RESULTS_TABLES, '空间溢出权重矩阵.xlsx')
FILE_Q2_DISTANCE = os.path.join(RESULTS_TABLES, '区域距离矩阵.xlsx')
FILE_Q2_OPTIMIZATION = os.path.join(RESULTS_TABLES, 'optimization_result.npz')
FILE_Q2_PREPROCESS = os.path.join(RESULTS_TABLES, 'preprocess_data.npz')

# 问题3结果文件
FILE_Q3_TABLE_A = os.path.join(RESULTS_Q3, '表A_调度前后峰谷差对比.xlsx')
FILE_Q3_TABLE_B = os.path.join(RESULTS_Q3, '表B_过载风险评估.xlsx')
FILE_Q3_BEFORE_ANALYSIS = os.path.join(RESULTS_Q3, '表_调度前分析.xlsx')
FILE_Q3_MERGED_DATA = os.path.join(RESULTS_Q3, 'merged_data.pkl')
FILE_Q3_PREPROCESS = os.path.join(RESULTS_Q3, 'preprocess_data.npz')
FILE_Q3_DISPATCH_UNIFORM = os.path.join(RESULTS_Q3, 'dispatch_uniform.pkl')
FILE_Q3_DISPATCH_WATERFILL = os.path.join(RESULTS_Q3, 'dispatch_waterfill.pkl')
FILE_Q3_DISPATCH_COMPARE = os.path.join(RESULTS_Q3, 'dispatch_compare.npz')

# 问题4结果文件
FILE_Q4_FINAL_RESULT = os.path.join(RESULTS_Q4, 'problem4_final_result.xlsx')
FILE_Q4_DEMAND = os.path.join(RESULTS_Q4, '未来需求推演.xlsx')
FILE_Q4_HEALTH = os.path.join(RESULTS_Q4, '健康度评价.xlsx')
FILE_Q4_EXPANSION = os.path.join(RESULTS_Q4, '动态扩容方案.xlsx')
FILE_Q4_PRIORITY = os.path.join(RESULTS_Q4, '扩容优先级.xlsx')
FILE_Q4_CAPACITY = os.path.join(RESULTS_Q4, '有效容量分析.xlsx')
FILE_Q4_SCENARIO = os.path.join(RESULTS_Q4, '情景对比汇总.xlsx')
FILE_Q4_PLAN = os.path.join(RESULTS_Q4, '未来三年扩展规划.xlsx')

# ═══════════════════════════════════════════════════════════════
# 自动创建必要目录
# ═══════════════════════════════════════════════════════════════
for _dir in [DATA_PROCESSED, MODELS_DIR, RESULTS_DIR,
             RESULTS_FIGURES, RESULTS_TABLES, RESULTS_LOGS,
             RESULTS_Q3, RESULTS_Q4]:
    os.makedirs(_dir, exist_ok=True)
