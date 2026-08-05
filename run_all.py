"""
统一运行入口
------------
按数学建模问题顺序运行全部代码。
运行前自动检查数据完整性。
"""

import subprocess
import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

PYTHON = sys.executable

STEPS = [
    # ── 数据检查 ──
    ('数据完整性检查', 'utils/check_data.py'),

    # ── 问题1：充电需求分析与预测 ──
    ('Q1-1 数据清洗',     'src/problem1/preprocess/data_clean.py'),
    ('Q1-2 区域聚类',     'src/problem1/analysis/cluster_analysis.py'),
    ('Q1-3 XGBoost模型',  'src/problem1/model/xgboost_model.py'),
    ('Q1-4 SHAP解释',     'src/problem1/model/shap_analysis.py'),
    ('Q1-5 预测汇总',     'src/problem1/main.py'),

    # ── 问题2：充电桩优化配置 ──
    ('Q2-1 供需分析',     'src/problem2/problem2_data.py'),
    ('Q2-2 NSGA-II优化',  'src/problem2/problem2_optimize.py'),
    ('Q2-3 结果汇总',     'src/problem2/problem2_result.py'),

    # ── 问题3：分时电价调度 ──
    ('Q3-1 调度前分析',   'src/problem3/problem3_data.py'),
    ('Q3-2 负荷转移',     'src/problem3/problem3_solve.py'),
    ('Q3-3 效果评估',     'src/problem3/problem3_result.py'),

    # ── 问题4：生命周期动态扩展规划 ──
    ('Q4 动态扩展规划',   'src/problem4/problem4_main.py'),
]


def main():
    print("=" * 60)
    print("城市新能源公共充电网络智能规划与调度")
    print("2026年数学建模B题 — 全流程运行")
    print("=" * 60)

    failed = []
    for name, script in STEPS:
        print(f"\n{'─' * 60}")
        print(f">>> {name}")
        print(f"    {script}")
        print(f"{'─' * 60}")

        script_path = os.path.join(ROOT, script)
        script_dir = os.path.dirname(script_path)
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        result = subprocess.run([PYTHON, script_path], cwd=script_dir, env=env)

        if result.returncode != 0:
            print(f"\n[FAIL] {name} (exit code {result.returncode})")
            failed.append(name)
            break
        else:
            print(f"\n[OK] {name}")

    print("\n" + "=" * 60)
    if failed:
        print(f"[FAIL] Stopped at: {failed}")
    else:
        print("[OK] All steps completed!")
        print(f"\n输出文件位置:")
        print(f"  问题1预测: results/prediction_result.xlsx")
        print(f"  问题2配置: results/tables/")
        print(f"  问题3评估: results/q3_output/")
        print(f"  问题4规划: results/q4_output/")
        print(f"  可视化图: results/figures/")
    print("=" * 60)

    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
