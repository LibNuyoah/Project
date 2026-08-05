"""
数据完整性检查工具
-----------------
启动项目前检查所有官方附件是否存在、Sheet 是否正确。
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

EXIT_CODE = 0


def check(msg, condition):
    global EXIT_CODE
    if condition:
        print(f"  [OK] {msg}")
    else:
        print(f"  [FAIL] {msg}")
        EXIT_CODE = 1


def main():
    print("=" * 60)
    print("数据完整性检查")
    print("=" * 60)

    # ── 1. data/raw 目录存在 ──
    print("\n[1] 检查 data/raw 目录...")
    check("data/raw 目录存在", os.path.isdir(DATA_RAW))

    # ── 2. 附件1-5 存在 ──
    print("\n[2] 检查官方附件文件...")
    attachments = {
        '附件1': FILE_ATTACHMENT1,
        '附件2': FILE_ATTACHMENT2,
        '附件3': FILE_ATTACHMENT3,
        '附件4': FILE_ATTACHMENT4,
        '附件5': FILE_ATTACHMENT5,
    }
    for name, path in attachments.items():
        check(f"{name} 存在", os.path.isfile(path))

    # ── 3. 附件2 Sheet 检查 ──
    print("\n[3] 检查附件2 Sheet...")
    if os.path.isfile(FILE_ATTACHMENT2):
        try:
            xl2 = pd.ExcelFile(FILE_ATTACHMENT2)
            sheets2 = xl2.sheet_names
            check(f"工作日Sheet: '{SHEET_ATTACHMENT2_WEEKDAY}'",
                  SHEET_ATTACHMENT2_WEEKDAY in sheets2)
            check(f"周末Sheet: '{SHEET_ATTACHMENT2_WEEKEND}'",
                  SHEET_ATTACHMENT2_WEEKEND in sheets2)
            # 尝试读取
            df_wd = pd.read_excel(FILE_ATTACHMENT2, sheet_name=SHEET_ATTACHMENT2_WEEKDAY)
            df_we = pd.read_excel(FILE_ATTACHMENT2, sheet_name=SHEET_ATTACHMENT2_WEEKEND)
            check(f"工作日数据: {df_wd.shape[0]}行×{df_wd.shape[1]}列",
                  df_wd.shape[0] >= 10 and df_wd.shape[1] >= 24)
            check(f"周末数据: {df_we.shape[0]}行×{df_we.shape[1]}列",
                  df_we.shape[0] >= 10 and df_we.shape[1] >= 24)
        except Exception as e:
            check(f"读取附件2: {e}", False)

    # ── 4. 附件3 Sheet 检查 ──
    print("\n[4] 检查附件3 Sheet...")
    if os.path.isfile(FILE_ATTACHMENT3):
        try:
            xl3 = pd.ExcelFile(FILE_ATTACHMENT3)
            sheets3 = xl3.sheet_names
            check(f"工作日Sheet: '{SHEET_ATTACHMENT3_WEEKDAY}'",
                  SHEET_ATTACHMENT3_WEEKDAY in sheets3)
            check(f"周末Sheet: '{SHEET_ATTACHMENT3_WEEKEND}'",
                  SHEET_ATTACHMENT3_WEEKEND in sheets3)
            df_wd3 = pd.read_excel(FILE_ATTACHMENT3, sheet_name=SHEET_ATTACHMENT3_WEEKDAY)
            df_we3 = pd.read_excel(FILE_ATTACHMENT3, sheet_name=SHEET_ATTACHMENT3_WEEKEND)
            check(f"工作日数据: {df_wd3.shape[0]}行×{df_wd3.shape[1]}列",
                  df_wd3.shape[0] >= 10 and df_wd3.shape[1] >= 24)
            check(f"周末数据: {df_we3.shape[0]}行×{df_we3.shape[1]}列",
                  df_we3.shape[0] >= 10 and df_we3.shape[1] >= 24)
        except Exception as e:
            check(f"读取附件3: {e}", False)

    # ── 结果 ──
    print("\n" + "=" * 60)
    if EXIT_CODE == 0:
        print("[OK] All checks passed.")
    else:
        print("[FAIL] Errors found, check data/raw/ directory.")
    print("=" * 60)

    return EXIT_CODE


if __name__ == '__main__':
    sys.exit(main())
