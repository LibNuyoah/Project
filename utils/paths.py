"""
项目路径管理模块
---------------
统一管理所有项目路径。从任意位置 import 均能正确定位。

用法:
    from utils.paths import DATA_RAW, FILE_ATTACHMENT1, RESULT_DIR
"""

import os

# ── 项目根目录 ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── 数据目录 ──
DATA_RAW = os.path.join(PROJECT_ROOT, 'data', 'raw')
DATA_PROCESSED = os.path.join(PROJECT_ROOT, 'data', 'processed')

# ── 模型目录（仅存放训练好的 .pkl 等二进制文件）──
MODEL_DIR = os.path.join(PROJECT_ROOT, 'model')

# ── 输出目录 ──
RESULT_DIR = os.path.join(PROJECT_ROOT, 'result')
RESULT_FIGURES = os.path.join(RESULT_DIR, 'figures')
RESULT_TABLES = os.path.join(RESULT_DIR, 'tables')
RESULT_Q3 = os.path.join(RESULT_DIR, 'q3_output')

# ── 文档目录 ──
DOCS_DIR = os.path.join(PROJECT_ROOT, 'docs')

# ── 各问题源码目录 ──
SRC_Q1 = os.path.join(PROJECT_ROOT, 'src', 'question1')
SRC_Q2 = os.path.join(PROJECT_ROOT, 'src', 'question2')
SRC_Q3 = os.path.join(PROJECT_ROOT, 'src', 'question3')
SRC_Q4 = os.path.join(PROJECT_ROOT, 'src', 'question4')

# ═══════════════════════════════════════════════════════════
# 官方原始附件（不可改名、不可拆分）
# 附件2和附件3各含两个sheet，通过 sheet_name 参数区分 工作日/周末
# ═══════════════════════════════════════════════════════════

FILE_ATTACHMENT1 = os.path.join(
    DATA_RAW, '附件 1 市主城区 10 个典型区域基础数据.xlsx'
)
FILE_ATTACHMENT2 = os.path.join(
    DATA_RAW, '附件2 市主城区 10 区域分时段充电车次.xlsx'
)
FILE_ATTACHMENT3 = os.path.join(
    DATA_RAW, '附件3 市主城区 10 区域分时段充电负荷.xlsx'
)
FILE_ATTACHMENT4 = os.path.join(
    DATA_RAW, '附件4 市主城区 10 个区域分时段电网最大允许负荷数据.xlsx'
)
FILE_ATTACHMENT5 = os.path.join(
    DATA_RAW, '附件5  基础参数补充.docx'
)

# ── 附件2 Sheet 名称 ──
SHEET_ATTACHMENT2_WEEKDAY = '工作日分时段充电车次数据'
SHEET_ATTACHMENT2_WEEKEND = '周末充电车次数据'

# ── 附件3 Sheet 名称 ──
SHEET_ATTACHMENT3_WEEKDAY = '工作日分时段充电负荷数据'
SHEET_ATTACHMENT3_WEEKEND = '周末充电负荷数据（修改后）'

# ── 处理后数据 ──
FILE_CLEAN_DATA = os.path.join(DATA_PROCESSED, 'clean_data.xlsx')

# ── 模型文件 ──
FILE_XGBOOST_MODEL = os.path.join(MODEL_DIR, 'xgboost_model.pkl')

# ── 结果文件 ──
FILE_PREDICTION_RESULT = os.path.join(RESULT_DIR, 'prediction_result.xlsx')
FILE_CLUSTER_RESULT = os.path.join(RESULT_TABLES, 'cluster_result.xlsx')

# ── 自动创建必要目录 ──
for _dir in [DATA_PROCESSED, MODEL_DIR, RESULT_DIR,
             RESULT_FIGURES, RESULT_TABLES, RESULT_Q3]:
    os.makedirs(_dir, exist_ok=True)
