# run_all.py 修复说明

**日期**: 2026-08-05  
**修复人**: Claude Code

---

## 问题背景

`run_all.py` 在项目根目录运行时，Q2 脚本（`problem2_data.py`）报错：
```
FileNotFoundError: '附件 1 市主城区 10 个典型区域基础数据.xlsx'
```

## 根因

1. **路径问题**：`run_all.py` 第 53 行使用 `cwd=ROOT`，所有脚本工作目录被强制设为项目根目录。但 Q2 脚本使用裸相对路径（如 `'附件 1...xlsx'`），数据文件实际在 `src/question2/` 下。

2. **GBK 编码问题**：多个脚本的 `print` 语句包含 Unicode 符号（`✅❌✓✗`），在 Windows GBK 控制台下触发 `UnicodeEncodeError`。

## 修改文件

| 文件 | 修改内容 |
|------|----------|
| `run_all.py` | (1) 子进程 `cwd` 改为脚本所在目录 (2) `✅❌` → `[OK]` `[FAIL]` (3) 添加 `PYTHONIOENCODING=utf-8` 环境变量 |
| `utils/check_data.py` | `✅❌` → `[OK]` `[FAIL]` |
| `src/question2/problem2_result.py` | `✓✗` → `OK` `MISS` |

## 关键修改

### run_all.py 第 53-55 行

```python
# 修改前
result = subprocess.run([PYTHON, script], cwd=ROOT)

# 修改后
script_path = os.path.join(ROOT, script)
script_dir = os.path.dirname(script_path)
env = os.environ.copy()
env['PYTHONIOENCODING'] = 'utf-8'
result = subprocess.run([PYTHON, script_path], cwd=script_dir, env=env)
```

每个脚本在自己的目录下运行，文件路径不再错乱。

## 备份位置

```
docs/backup/
├── original/          # 修改前的原版（git 导出）
│   ├── run_all.py
│   ├── check_data.py
│   └── problem2_result.py
├── modified/          # 修改后的版本
│   ├── run_all.py
│   ├── check_data.py
│   └── problem2_result.py
├── changes.diff       # git diff 补丁文件
└── 本文档
```

## 还原方法

```bash
cd C:\Users\19045\Project
git checkout -- run_all.py utils/check_data.py src/question2/problem2_result.py
```

或直接复制 `docs/backup/original/` 下的文件覆盖。
