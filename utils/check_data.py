"""
数据完整性检查工具
-----------------
检查所有官方附件是否存在、Sheet 是否正确、数据规模与字段。
"""

import os
import sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.paths import (
    FILE_ATTACHMENT1, FILE_ATTACHMENT2, FILE_ATTACHMENT3,
    FILE_ATTACHMENT4, FILE_ATTACHMENT5,
    SHEET_ATTACHMENT2_WEEKDAY, SHEET_ATTACHMENT2_WEEKEND,
    SHEET_ATTACHMENT3_WEEKDAY, SHEET_ATTACHMENT3_WEEKEND,
    DATA_RAW
)


def main():
    print("=" * 60)
    print("数据检查开始")
    print("=" * 60)

    check_passed = True

    # ── 附件1 ──
    print("\n附件1：")
    print("文件路径：")
    print(FILE_ATTACHMENT1)
    if os.path.isfile(FILE_ATTACHMENT1):
        df1 = pd.read_excel(FILE_ATTACHMENT1)
        df1_valid = df1.iloc[:10]
        print("\n数据规模：")
        print(df1_valid.shape)
        print("\n字段：")
        for col in df1_valid.columns:
            print(col)
        print("\n缺失值：")
        print(df1_valid.isnull().sum().sum())
        print("\n重复值：")
        print(df1_valid.duplicated().sum())
    else:
        print("\n[FAIL] 文件不存在")
        check_passed = False

    # ── 附件2 ──
    print("\n\n附件2：")
    print("文件路径：")
    print(FILE_ATTACHMENT2)
    if os.path.isfile(FILE_ATTACHMENT2):
        xl2 = pd.ExcelFile(FILE_ATTACHMENT2)
        print("\nSheet列表：")
        for s in xl2.sheet_names:
            print(f"  - {s}")
        df2_wd = pd.read_excel(FILE_ATTACHMENT2, sheet_name=SHEET_ATTACHMENT2_WEEKDAY)
        df2_we = pd.read_excel(FILE_ATTACHMENT2, sheet_name=SHEET_ATTACHMENT2_WEEKEND)
        print("\n数据规模：")
        print(f"  工作日: {df2_wd.shape}")
        print(f"  周末: {df2_we.shape}")
        print("\n缺失值：")
        print(f"  工作日: {df2_wd.isnull().sum().sum()}")
        print(f"  周末: {df2_we.isnull().sum().sum()}")
    else:
        print("\n[FAIL] 文件不存在")
        check_passed = False

    # ── 附件3 ──
    print("\n\n附件3：")
    print("文件路径：")
    print(FILE_ATTACHMENT3)
    if os.path.isfile(FILE_ATTACHMENT3):
        xl3 = pd.ExcelFile(FILE_ATTACHMENT3)
        print("\nSheet列表：")
        for s in xl3.sheet_names:
            print(f"  - {s}")
        df3_wd = pd.read_excel(FILE_ATTACHMENT3, sheet_name=SHEET_ATTACHMENT3_WEEKDAY)
        df3_we = pd.read_excel(FILE_ATTACHMENT3, sheet_name=SHEET_ATTACHMENT3_WEEKEND)
        print("\n数据规模：")
        print(f"  工作日: {df3_wd.shape}")
        print(f"  周末: {df3_we.shape}")
        print("\n缺失值：")
        print(f"  工作日: {df3_wd.isnull().sum().sum()}")
        print(f"  周末: {df3_we.isnull().sum().sum()}")
    else:
        print("\n[FAIL] 文件不存在")
        check_passed = False

    # ── 附件4 ──
    print("\n\n附件4：")
    print("文件路径：")
    print(FILE_ATTACHMENT4)
    if os.path.isfile(FILE_ATTACHMENT4):
        df4 = pd.read_excel(FILE_ATTACHMENT4)
        print("\n数据规模：")
        print(df4.shape)
        print("\n字段：")
        for col in df4.columns:
            print(col)
        print("\n缺失值：")
        print(df4.isnull().sum().sum())
    else:
        print("\n[FAIL] 文件不存在")
        check_passed = False

    # ── 附件5 ──
    print("\n\n附件5：")
    print("文件路径：")
    print(FILE_ATTACHMENT5)
    if os.path.isfile(FILE_ATTACHMENT5):
        print("\n文件存在: 是")
    else:
        print("\n文件存在: 否")
        check_passed = False

    # ── 结果 ──
    if check_passed:
        print("\n\n检查通过")

    print("\n" + "=" * 60)
    print("全部数据检查完成")
    print("=" * 60)


if __name__ == '__main__':
    main()
