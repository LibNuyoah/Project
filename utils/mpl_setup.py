"""
=============================================================================
matplotlib 中文字体统一配置
=============================================================================
使用方法：在任何使用 matplotlib 的脚本开头添加一行：

    from utils.mpl_setup import setup_chinese

    setup_chinese()

该函数会：
  1. 检测系统可用的中文字体（SimHei → Microsoft YaHei → DejaVu Sans 回退）
  2. 设置 Agg 后端（无 GUI 环境安全）
  3. 修复负号显示问题
  4. 清除字体缓存确保字体生效
=============================================================================
"""

import matplotlib
matplotlib.use('Agg')  # 非交互后端，服务器/脚本环境安全

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import warnings


def _get_available_chinese_font():
    """检测系统可用的中文字体，按优先级返回"""
    available = {f.name for f in fm.fontManager.ttflist}
    # 优先级：SimHei > Microsoft YaHei > STXihei > FangSong > DejaVu Sans
    candidates = ['SimHei', 'Microsoft YaHei', 'STXihei', 'STSong',
                  'FangSong', 'KaiTi', 'Microsoft JhengHei']
    for font in candidates:
        if font in available:
            return font
    return 'DejaVu Sans'


def setup_chinese():
    """配置 matplotlib 以正确显示中文"""
    font_name = _get_available_chinese_font()

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': [font_name, 'DejaVu Sans', 'Arial'],
        'axes.unicode_minus': False,
        'mathtext.fontset': 'stix',
    })

    # 清除字体缓存，强制重新扫描
    fm._load_fontmanager(try_read_cache=False)

    print(f'[matplotlib] 中文字体配置: {font_name}')
    return font_name


if __name__ == '__main__':
    setup_chinese()
    # 快速测试
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.set_title('中文字体测试 — 充电负荷预测')
    ax.set_xlabel('时间（小时）')
    ax.set_ylabel('负荷（千瓦）')
    fig.savefig('_font_test.png', dpi=100, bbox_inches='tight')
    plt.close()
    print('测试图片: _font_test.png')
