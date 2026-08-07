"""
区域功能特征加载器
-----------------
从聚类分析结果加载区域功能特征映射（单一数据源），
替代所有模块中的硬编码 REGION_TYPES。
"""

import pandas as pd
import os
import sys

# 缓存
_region_types = None
_region_names = {
    1: '宝塔山街道', 2: '南市街道', 3: '凤凰山街道',
    4: '桥沟街道', 5: '枣园街道', 6: '新城街道',
    7: '河庄坪镇', 8: '姚店镇（经开区）', 9: '万花山镇', 10: '真武洞街道（安塞）'
}


def get_region_types():
    """
    从聚类分析结果加载区域功能特征映射。
    若聚类结果不存在，返回基于数据的默认映射。
    """
    global _region_types
    if _region_types is not None:
        return _region_types

    # 尝试从聚类结果文件加载
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.path.insert(0, project_root)
    from utils.paths import FILE_CLUSTER_RESULT
    cluster_file = FILE_CLUSTER_RESULT

    if os.path.exists(cluster_file):
        df = pd.read_excel(cluster_file)
        _region_types = dict(zip(df['区域编号'], df['功能特征']))
    else:
        # 回退：基于聚类分析K=3结果的硬编码
        _region_types = {
            1: '老城核心区', 2: '老城核心区', 3: '城市新区',
            4: '老城核心区', 5: '城市新区', 6: '城市新区',
            7: '城郊/工业区', 8: '城郊/工业区', 9: '城郊/工业区',
            10: '城郊/工业区'
        }

    return _region_types


def get_region_names():
    """获取区域编号→名称映射"""
    return _region_names


# 导出快捷访问
REGION_TYPES = get_region_types()
REGION_NAMES = _region_names
