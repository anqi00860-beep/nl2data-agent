# -*- coding: utf-8 -*-
"""
任务8：智能数据分析助手（RAG）
====================================================================
整体流程：
    数据准备 → 数据预处理 → 字段元信息构建（知识库）→ 向量检索
    → 大模型问答 / Pandas 代码生成 → 图表展示 → 准确性评估

使用前准备：
    1. 把数据文件（.csv / .xlsx / .xls）放进本文件旁边的 data/ 文件夹；
    2. 在文件开头的“基础配置”里填上 API Key（或配置环境变量 DEEPSEEK_API_KEY）；
    3. 安装依赖：pip install -r requirements.txt；
    4. 运行本文件。

可选设置（见下方“基础配置”）：
    - QUESTIONS：自己出题；留空则让大模型根据字段自动出 5 题；
    - SHEET_NAME：Excel 有多个工作表时，指定读第几个（0 = 第一个）；
    - MERGE_KEY / MERGE_HOW：data 里放多个表时，按哪一列、用什么方式合并。
"""

import contextlib
import io
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==================== 基础配置 ====================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"          # 数据放在这里
OUT_DIR = BASE_DIR / "outputs"        # 结果、图表、报告自动生成到这里

# 【重要】两种填 API Key 的方式，二选一：
#   方式 A（最简单）：把下面引号里换成你的 Key，例如 "sk-abc123..."
#   方式 B（更规范）：这里留空，去 PyCharm 运行配置里加环境变量 DEEPSEEK_API_KEY
API_KEY = ""
API_KEY = API_KEY.strip() or os.getenv("DEEPSEEK_API_KEY", "").strip()

BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-flash"              # 官方模型：deepseek-flash（便宜快）/ deepseek-v4-pro（更强）
THINKING = "disabled"                 # "enabled"=深度思考，答案更准但更慢；"disabled"=更快
THINKING_EFFORT = "high"              # 仅 THINKING="enabled" 时生效：low / high / max

TOP_K = 4                             # 每次检索几条知识片段
SHEET_NAME = 0                        # Excel 读取第几个工作表

# 多表合并：data 里放多个表时，按这一列合并（只有一个表时不用管）
MERGE_KEY = "stkcd"
# 不同表里这一列可能叫别的名字，按顺序找，找到就用
MERGE_KEY_ALIASES = ("isssymbol", "issuerstockcode", "stockcode", "证券代码", "发行人股票代码")
MERGE_ON_YEAR = True                  # 两张表都有日期时，再按“年份”对齐（避免面板数据行数翻倍）
MERGE_YEAR_COL = "年份"
MERGE_HOW = "inner"                   # inner=只保留两边都有的；left=保留第一个表的全部

# 如果 data 里有这个文件，就直接用它作为分析面板（不再合并原始表）
PANEL_FILE = "公司年度面板.csv"

MAX_CODE_RETRY = 1                    # 生成代码执行失败后，让模型改几次

# 自己出题就填在这里；留空 [] 表示让大模型自动出题
# 下面这组题是按“公司-年度面板”设计的，对应谭之博(2012)的规模-融资来源框架
QUESTIONS = [
    "全样本的银行融资占比、债券融资占比的均值和中位数分别是多少？",
    "企业规模低于中位数的小企业，和大企业相比，银行融资占比、债券融资占比分别差多少？",
    "把企业按规模从低到高分五组，各组的银行融资占比、债券融资占比、资产负债率分别是多少？",
    "发债企业和未发债企业，在企业规模、资产负债率、盈利能力上有什么差异？",
    "2015 到 2025 年，发债企业数量和平均债券融资占比怎么变化？",
    "企业规模与银行融资占比的相关系数是多少？",
]
AUTO_QUESTION_COUNT = 5

# 控制台统一用 UTF-8，避免中文/符号在某些终端里乱码或报错
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 60)

_client = None


# ==================== 大模型调用 ====================
def get_client():
    """延迟创建客户端：这样没配 Key 时能给出友好提示，而不是一串英文报错。"""
    global _client
    if _client is None:
        if not API_KEY:
            raise RuntimeError(
                "还没有填 API Key。\n"
                '请打开本文件开头的“基础配置”，把  API_KEY = ""  改成  API_KEY = "sk-..."；\n'
                "或者配置环境变量 DEEPSEEK_API_KEY。保存后重新运行。"
            )
        _client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=180, max_retries=0)
    return _client


def chat(messages, json_mode=False, temperature=0.0, retries=3):
    """调用 DeepSeek，失败时自动重试。"""
    last_err = None
    for attempt in range(retries):
        try:
            kwargs = {
                "model": MODEL,
                "messages": messages,
                "temperature": temperature,
            }
            if THINKING == "enabled":
                kwargs["reasoning_effort"] = THINKING_EFFORT
                kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
            else:
                kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            resp = get_client().chat.completions.create(**kwargs)
            content = (resp.choices[0].message.content or "").strip()
            if content:
                return content
            last_err = RuntimeError("模型返回了空内容")
        except Exception as e:  # 网络抖动、限流等，重试
            last_err = e
            # Key 错、余额不足这类问题重试也没用，直接抛出
            if getattr(e, "status_code", None) in (400, 401, 402, 403, 404):
                break
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"调用大模型失败：{last_err}")


def parse_json(text):
    """模型偶尔会在 JSON 外面包 ```，这里做兼容。"""
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", t, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


# ==================== 1. 数据准备 ====================
def find_data_files():
    exts = {".csv", ".xlsx", ".xls"}
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    panel = DATA_DIR / PANEL_FILE
    if panel.exists():
        print(f"  检测到分析面板 {PANEL_FILE}，直接使用（原始表不再合并）")
        return [panel]

    files = sorted(
        p for p in DATA_DIR.iterdir()
        if p.suffix.lower() in exts and not p.name.startswith("~$")
    )
    if not files:
        raise FileNotFoundError(
            f"data 文件夹里没有数据文件：{DATA_DIR}\n"
            "请把 .csv / .xlsx / .xls 文件复制进去（第一行要是列名）。\n"
            "放多个表也可以，程序会用 MERGE_KEY 指定的那一列把它们合并。"
        )
    return files


CODE_TAILS = ("cd", "code", "id", "symbol")


def strip_prefix(name):
    """CSR_Finidx.Stkcd -> Stkcd；没有前缀就原样返回。"""
    s = str(name).strip()
    if "." in s:
        head, tail = s.split(".", 1)
        if tail:
            return tail
    return s


def is_code_like(name):
    """判断是不是“代码/编号”列：这类列必须按文本读，否则 000002 会变成 2。"""
    tail = re.split(r"[._]", str(name).lower())[-1]
    return tail.endswith(CODE_TAILS)


def is_unit_row(row):
    """识别 CSMAR 导出的“单位行”（没有单位 / 元 / 亿元 / 年 ...）。"""
    vals = [str(v).strip() for v in row.tolist()]
    unit_words = {"没有单位", "无单位", "单位", "元", "亿元", "万元", "千元",
                  "年", "%", "次", "股", "人", "户", "无", ""}
    hits = sum(1 for v in vals if v in unit_words or "单位" in v)
    return hits >= max(3, int(len(vals) * 0.5))


def clean_label(v):
    s = str(v).strip()
    return "" if s.lower() in ("nan", "none") else s


def load_table(path):
    """读一张表，返回 (DataFrame, 中文列名, 单位)。

    兼容 CSMAR 导出的三行表头：
        第 1 行：英文列名（Liscd, IssSymbol ...）
        第 2 行：中文列名（债券代码, 发行人股票代码 ...）
        第 3 行：单位（没有单位, 亿元 ...）
        第 4 行起才是数据
    也兼容普通只有一行表头的 Excel / CSV。
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        last_err = None
        for enc in ("utf-8-sig", "gbk", "utf-8"):
            try:
                df = pd.read_csv(path, encoding=enc)
                # 如果只读出一列，可能是分隔符不是逗号
                if df.shape[1] == 1:
                    for sep in ("\t", ";", "|"):
                        tmp = pd.read_csv(path, encoding=enc, sep=sep)
                        if tmp.shape[1] > 1:
                            return tmp, {}, {}
                if "stkcd" in df.columns:
                    df["stkcd"] = (df["stkcd"].astype(str)
                                   .str.replace(r"\.0$", "", regex=True)
                                   .str.strip().str.zfill(6))
                return df, {}, {}
            except UnicodeDecodeError as e:
                last_err = e
                continue
        raise RuntimeError(
            f"CSV 编码识别失败：{last_err}\n"
            "请在 Excel 里“另存为 → CSV UTF-8（逗号分隔）”后再试。"
        )

    preview = pd.read_excel(path, sheet_name=SHEET_NAME, header=None, nrows=3, dtype=str)
    if preview.shape[0] == 0:
        raise RuntimeError(f"{path.name} 是空表。")
    cols = [str(c).strip() for c in preview.iloc[0].tolist()]

    labels, units, skip = {}, {}, []
    if preview.shape[0] >= 3 and is_unit_row(preview.iloc[2]):
        # 中文名行 + 单位行
        labels = {c: clean_label(v) for c, v in zip(cols, preview.iloc[1].tolist())}
        units = {c: clean_label(v) for c, v in zip(cols, preview.iloc[2].tolist())}
        skip = [1, 2]
    elif preview.shape[0] >= 2 and is_unit_row(preview.iloc[1]):
        units = {c: clean_label(v) for c, v in zip(cols, preview.iloc[1].tolist())}
        skip = [1]

    dtype_map = {c: str for c in cols if is_code_like(c)}
    df = pd.read_excel(
        path, sheet_name=SHEET_NAME, header=0,
        skiprows=skip or None, dtype=dtype_map or None,
    )
    # 去掉 CSR_Finidx. 这类前缀，字段名更适合写代码、写问题
    df.columns = [strip_prefix(c) for c in df.columns]
    labels = {strip_prefix(k): v for k, v in labels.items()}
    units = {strip_prefix(k): v for k, v in units.items()}
    return df, labels, units


def find_column(df, aliases):
    """按候选名字找列（忽略大小写），aliases 越靠前优先级越高。"""
    lower = {str(c).strip().lower(): c for c in df.columns}
    for a in aliases:
        key = str(a).strip().lower()
        if key in lower:
            return lower[key]
    return None


def find_date_column(df):
    """找一张表里用来对齐年份的日期列，并把它转成日期类型。"""
    hints = ("ipodt", "issuedt", "issue_date", "accper", "listdt", "matdt", "date", "日期", "时间")
    for hint in hints:
        for c in df.columns:
            if hint in str(c).lower():
                try:
                    s = pd.to_datetime(df[c], errors="coerce")
                except Exception:
                    continue
                if s.notna().mean() > 0.8:
                    df[c] = s
                    return c
    return None


def normalize_key(s):
    """把合并列统一成干净的字符串：去空格、去 .0、去千分位逗号。"""
    out = s.astype(str).str.strip()
    out = out.str.replace(r"\.0$", "", regex=True)
    out = out.str.replace(",", "", regex=False)
    out = out.replace({"nan": np.nan, "None": np.nan, "": np.nan, "<NA>": np.nan})
    return out


def load_dataset():
    """读取 data 里的数据。

    - 只有一个文件：直接返回；
    - 多个文件：按 MERGE_KEY 合并；如果两边都有日期列，再按“年份”对齐。
    返回：(DataFrame, 数据说明, 合并提示列表, 中文列名, 单位)
    """
    files = find_data_files()
    frames, names, labels_all, units_all = [], [], {}, {}
    for p in files:
        d, labels, units = load_table(p)
        if d is None or d.empty:
            raise RuntimeError(f"{p.name} 读出来是空表，请检查文件内容。")
        d, _ = clean_df(d)
        frames.append(d)
        names.append(p.stem)
        for k, v in labels.items():
            if v:
                labels_all.setdefault(k, v)
        for k, v in units.items():
            if v:
                units_all.setdefault(k, v)

    if len(frames) == 1:
        notes = []
        if files[0].name == PANEL_FILE:
            notes.append(f"使用分析面板：{PANEL_FILE}（变量定义与样本筛选见 outputs/数据处理说明.md）")
        return frames[0], files[0].name, notes, labels_all, units_all

    key = MERGE_KEY
    aliases = (MERGE_KEY,) + tuple(MERGE_KEY_ALIASES)
    notes = []
    for d, name in zip(frames, names):
        col = find_column(d, aliases)
        if col is None:
            raise RuntimeError(
                f"{name} 里没有找到合并列（尝试过：{'、'.join(aliases)}）。\n"
                "这张表的列有：" + "、".join(map(str, d.columns)) + "\n"
                "请把代码里的 MERGE_KEY 改成实际列名。"
            )
        if col != key:
            d.rename(columns={col: key}, inplace=True)
            notes.append(f"{name}：合并列用 “{col}”")
            if labels_all.get(col):
                labels_all.setdefault(key, labels_all[col])
        d[key] = normalize_key(d[key])
        miss = int(d[key].isna().sum())
        if miss:
            notes.append(f"{name} 有 {miss:,} 行没有 {col}，无法参与合并")

    # 一边是数字、一边是带前导零的文本时（1 vs 000001），统一补零
    if all(d[key].dropna().str.fullmatch(r"\d+").all() for d in frames):
        width = max((d[key].dropna().str.len().max() or 0) for d in frames)
        if 1 <= width <= 10:
            for d in frames:
                d[key] = d[key].str.zfill(int(width))

    # 找每张表的日期列，用于按年份对齐
    date_cols = [find_date_column(d) for d in frames]

    merge_keys = [key]
    if MERGE_ON_YEAR and all(date_cols):
        yname = MERGE_YEAR_COL
        if any(yname in d.columns for d in frames):
            yname = yname + "_合并"
        for d, dc in zip(frames, date_cols):
            d[yname] = d[dc].dt.year
        if all(d[yname].notna().mean() > 0.8 for d in frames):
            merge_keys.append(yname)
            notes.append("按年份对齐：" + "，".join(
                f"{n} 用 {dc}" for n, dc in zip(names, date_cols)))
        else:
            for d in frames:
                d.drop(columns=[yname], inplace=True)

    # 右边的表如果是“公司-年度”面板，这里只留下能和债券对上的那一行
    for i, (d, name) in enumerate(zip(frames[1:], names[1:]), start=1):
        before = len(d)
        if len(merge_keys) == 1 and date_cols[i] is not None:
            d.sort_values(date_cols[i], inplace=True)  # 升序，保留最后一条=最新
        d.drop_duplicates(subset=merge_keys, keep="last", inplace=True)
        if len(d) < before:
            notes.append(
                f"{name} 是面板数据：按 {'+'.join(merge_keys)} 去重 "
                f"{before:,} → {len(d):,} 行（每个组合只留一条）"
            )

    merged = frames[0]
    used = set(merged.columns)
    for d, name in zip(frames[1:], names[1:]):
        clash = [c for c in d.columns if c in used and c not in merge_keys]
        if clash:
            for c in clash:
                d.rename(columns={c: f"{name}_{c}"}, inplace=True)
            notes.append(f"{name} 里重名的列已加前缀：" + "、".join(f"{name}_{c}" for c in clash))
        left_rows = len(merged)
        merged = merged.merge(d, on=merge_keys, how=MERGE_HOW, suffixes=("", "_dup"))
        notes.append(f"合并 {name}：{left_rows:,} 行 + {len(d):,} 行 → {len(merged):,} 行")

    if merged.empty:
        sample_a = "、".join(map(str, frames[0][key].dropna().unique()[:3]))
        sample_b = "、".join(map(str, frames[1][key].dropna().unique()[:3]))
        raise RuntimeError(
            f"按 {' + '.join(merge_keys)} 合并后一行都没有。\n"
            f"{names[0]} 的样例：{sample_a}\n"
            f"{names[1]} 的样例：{sample_b}\n"
            "请检查两边的股票代码格式是否一致（比如一个带 .SZ 后缀、一个不带），"
            "或者把 MERGE_ON_YEAR 改成 False 再试。"
        )

    if MERGE_HOW == "inner" and len(merged) < len(frames[0]) * 0.5:
        notes.append('注意：合并后只保留了不到一半的记录。想保留第一个表的全部，把 MERGE_HOW 改成 "left"。')
    if MERGE_HOW == "inner" and len(merged) < len(frames[0]) * 0.2:
        notes.append("匹配率偏低：右表的公司范围可能比左表窄（比如不含银行等金融机构），这属于数据口径问题，不是代码错。")

    notes.append(f"合并结果：{len(merged):,} 行 × {len(merged.columns)} 列，"
                 f"涉及 {merged[key].nunique():,} 个 {key}")
    notes.append(f"合并方式：{MERGE_HOW}（inner=只保留两边都能对上的；left=保留第一个表全部）")

    label = " + ".join(p.name for p in files)
    return merged, label, notes, labels_all, units_all


def clean_df(df):
    """轻量清洗：去空行空列、规范列名、去掉字符串首尾空格。"""
    before = df.shape
    df = df.copy()
    df.columns = [str(c).replace("\n", " ").replace("\r", " ").strip() for c in df.columns]

    seen, new_cols = {}, []
    for i, c in enumerate(df.columns):
        name = c if c and c.lower() != "nan" else f"列{i + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        new_cols.append(name)
    df.columns = new_cols

    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")

    for col in df.columns:
        s = df[col]
        if not pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_datetime64_any_dtype(s):
            df[col] = s.map(lambda v: v.strip() if isinstance(v, str) else v)
            # 只有空格、空字符串的单元格，统一当成缺失值
            df[col] = df[col].replace(r"^\s*$", np.nan, regex=True)

    return df.reset_index(drop=True), before


# ==================== 2. 字段元信息（知识库） ====================
def fmt_num(x):
    if x is None or (isinstance(x, float) and np.isnan(x)) or pd.isna(x):
        return "空"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if abs(v) >= 1000:
        return f"{v:,.2f}"
    return f"{v:.4f}".rstrip("0").rstrip(".")


def is_id_like(col, s):
    """判断是不是“编号/序号”这类只用来区分记录的列。"""
    name = str(col).lower()
    if any(k in name for k in ("编号", "序号", "学号", "工号")):
        return True
    return bool(re.search(r"(^|_)(id|no|index)$", name))


def build_knowledge_base(df, data_label, labels=None, units=None):
    """把“表结构 + 每列的统计信息 + 取值样例”写成一条条文本，作为知识库。"""
    labels = labels or {}
    units = units or {}
    docs = []
    n_rows, n_cols = df.shape
    col_list = "、".join(map(str, df.columns))

    docs.append({
        "title": "数据集总览",
        "text": (
            f"数据集名称：{data_label}。共 {n_rows} 行（条记录），{n_cols} 列（个字段）。"
            f"全部字段名：{col_list}。缺失值总数：{int(df.isna().sum().sum())}。"
            f"每一行代表一条记录。"
        ),
    })

    num_cols, cat_cols, date_cols = [], [], []
    for col in df.columns:
        s = df[col]
        n_missing = int(s.isna().sum())
        n_unique = int(s.nunique(dropna=True))
        info = [
            f"字段名：{col}",
            f"数据类型：{s.dtype}",
            f"非空数量：{int(s.notna().sum())}",
            f"缺失数量：{n_missing}",
            f"不同取值个数：{n_unique}",
        ]
        cn = labels.get(str(col), "")
        unit = units.get(str(col), "")
        if cn and cn != str(col):
            info.append(f"中文名：{cn}")
        if unit and unit not in ("没有单位", "无单位", "无", ""):
            info.append(f"单位：{unit}")

        if pd.api.types.is_bool_dtype(s):
            cat_cols.append(str(col))
            info.append("取值分布：" + "；".join(f"{k}（{v} 条）" for k, v in s.value_counts().items()))
        elif pd.api.types.is_numeric_dtype(s):
            num_cols.append(str(col))
            if is_id_like(col, s):
                info.append("说明：这是编号/标识列，每行一个唯一值，只用来区分记录，不要对它求平均或求和。")
            else:
                desc = s.describe()
                info += [
                    f"最小值：{fmt_num(desc['min'])}",
                    f"最大值：{fmt_num(desc['max'])}",
                    f"平均值：{fmt_num(desc['mean'])}",
                    f"中位数：{fmt_num(s.median())}",
                    f"标准差：{fmt_num(desc['std'])}",
                    f"合计：{fmt_num(s.sum())}",
                    f"25%分位数：{fmt_num(s.quantile(0.25))}",
                    f"75%分位数：{fmt_num(s.quantile(0.75))}",
                ]
        elif pd.api.types.is_datetime64_any_dtype(s):
            date_cols.append(str(col))
            info += [f"最早时间：{s.min()}", f"最晚时间：{s.max()}"]
        else:
            cat_cols.append(str(col))
            vc = s.dropna().astype(str).value_counts().head(5)
            info.append("出现最多的取值：" + "；".join(f"{k}（{v} 条）" for k, v in vc.items()))

        samples = [str(v) for v in s.dropna().unique()[:3]]
        info.append("样例值：" + "、".join(samples))
        docs.append({"title": f"字段：{col}", "text": "。".join(info) + "。"})

    docs.append({
        "title": "字段分类与分析角度",
        "text": (
            f"数值型字段（可以求平均值、求和、最大最小、分组统计）：{'、'.join(num_cols) or '无'}。"
            f"分类/文本型字段（可以分组、筛选、计数、算占比）：{'、'.join(cat_cols) or '无'}。"
            f"时间型字段（可以看趋势）：{'、'.join(date_cols) or '无'}。"
            "常见分析：按分类字段分组后对数值字段求平均/求和/计数；"
            "按条件筛选记录再计数；按数值排序取前 N 名；计算某类记录的占比。"
        ),
    })
    return docs


class VectorStore:
    """轻量向量库：TF-IDF 把文本变成向量，用余弦相似度做检索。

    说明：TF-IDF 也是把文本表示成向量，只是不用下载深度学习模型，
    在没网的教室电脑上也能稳定跑起来（作业演示更稳）。
    """

    def __init__(self, docs):
        self.docs = docs
        texts = [f"{d['title']}。{d['text']}" for d in docs]
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3))
        self.matrix = self.vectorizer.fit_transform(texts)

    def search(self, query, k=TOP_K):
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.matrix)[0]
        order = np.argsort(sims)[::-1][:k]
        return [(self.docs[i], float(sims[i])) for i in order]


# ==================== 3. 出题 ====================
def compact_schema(df, labels=None):
    labels = labels or {}
    lines = []
    for col in df.columns:
        s = df[col]
        sample = "、".join(str(v) for v in s.dropna().unique()[:3])
        cn = labels.get(str(col), "")
        cn = f"{cn}，" if cn else ""
        lines.append(f"- {col}（{cn}{s.dtype}，{s.nunique(dropna=True)} 种取值，例如：{sample}）")
    return "\n".join(lines)


def fallback_questions(df):
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])][:2]
    cat = [c for c in df.columns
           if not pd.api.types.is_numeric_dtype(df[c])
           and not pd.api.types.is_datetime64_any_dtype(df[c])
           and 1 < df[c].nunique(dropna=True) <= 50][:1]
    qs = [f"这个数据集一共有多少条记录？"]
    if num:
        qs.append(f"{num[0]}的平均值是多少？")
    if num and cat:
        qs.append(f"按{cat[0]}分组，{num[0]}的平均值分别是多少？")
    if num:
        qs.append(f"{num[0]}最高的前 5 条记录，对应的关键字段是什么？")
    if cat:
        vc = df[cat[0]].value_counts(normalize=True)
        if len(vc):
            qs.append(f"{cat[0]}中“{vc.index[0]}”这一类占比是多少？")
    return qs[:5]


def generate_questions(df, labels=None):
    prompt = f"""下面是一个数据集的字段说明（括号里是中文名）：

{compact_schema(df, labels)}

请为这个数据集设计 {AUTO_QUESTION_COUNT} 个中文自然语言问题，覆盖不同类型：
1 个统计类（平均值 / 总和 / 计数）；
1 个筛选类（满足某个条件的记录有多少 / 有哪些）；
1 个聚合类（按某个字段分组后统计）；
1 个排序类（最高 / 最低 / 最多 / 最少 / 前 N 名）；
1 个占比或对比类。

要求：
- 必须使用上面真实存在的字段名；
- 问题要具体，能通过对原始表做 pandas 计算得到答案；
- 不要把答案写进问题里；
- 只输出 JSON，格式：{{"questions": ["问题1", "问题2", "问题3", "问题4", "问题5"]}}"""

    print("  正在让大模型根据字段自动出题 ...")
    try:
        data = parse_json(chat(
            [{"role": "system", "content": "你是数据分析出题老师，只输出 JSON。"},
             {"role": "user", "content": prompt}],
            json_mode=True,
        ))
        qs_raw = (data or {}).get("questions", [])
        qs = [str(q).strip() for q in qs_raw if str(q).strip()] if isinstance(qs_raw, list) else []
        if len(qs) >= 3:
            return qs[:AUTO_QUESTION_COUNT]
    except Exception as e:
        print(f"  自动出题失败（{e}），改用通用问题。")

    return fallback_questions(df)


# ==================== 4. 问答 + 代码生成 ====================
SYSTEM_RAG = "你是一个严谨的数据分析助手。你只依据提供的数据字典和真实执行结果回答，绝不编造数字。"


def build_rag_messages(df, hits, question, labels=None):
    labels = labels or {}
    context = "\n\n".join(
        f"[片段{i}] {d['title']}：{d['text']}" for i, (d, _) in enumerate(hits, 1)
    )
    cols = "\n".join(
        f"- {c}（{labels.get(str(c), '') + '，' if labels.get(str(c)) else ''}{df[c].dtype}）"
        for c in df.columns
    )
    user = f"""下面是从数据知识库中检索到的信息：
{context}

数据表的全部字段（变量 df 就是这张表）：
{cols}

用户问题：{question}

请判断怎么回答这个问题：
- 只有“字段是什么意思、有哪些取值、有哪些类别”这类描述性问题，才直接回答，need_code 设为 false；
- 只要答案涉及具体数字（计数、平均、合计、占比、分组统计、排序、趋势），必须 need_code 设为 true 并写代码。
  即使知识库里已经有类似统计量，也要用代码在原始数据上重新算一遍，保证结果可核对。

写代码的要求：
1. 变量 df 是已经加载好的 DataFrame，直接使用；
2. 只能使用 pandas 和 numpy，不要读写文件、不要画图、不要联网；
3. 必须把最终结果赋值给变量 result（可以是数字、Series 或 DataFrame）；
4. 不要使用 print，不要输出多余内容；
5. 列名必须使用上面给出的真实字段名。

只输出一个 JSON 对象，不要输出别的文字：
{{"need_code": true 或 false, "answer": "文字回答（need_code 为 true 时留空）", "code": "pandas 代码（need_code 为 false 时留空）"}}"""
    return [{"role": "system", "content": SYSTEM_RAG}, {"role": "user", "content": user}]


def clean_code(code):
    if not code:
        return ""
    code = str(code).strip()
    code = re.sub(r"^```[a-zA-Z]*\s*", "", code)
    code = re.sub(r"\s*```$", "", code)
    return code.strip()


def execute_code(code, df):
    env = {"df": df.copy(), "pd": pd, "np": np}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(code, env)  # noqa: S102  本机运行模型生成的代码，仅用于作业演示
    return env.get("result"), buf.getvalue().strip()


def format_result(result, max_rows=30):
    if result is None:
        return ""
    if isinstance(result, pd.DataFrame):
        text = result.head(max_rows).to_string(index=False)
        if len(result) > max_rows:
            text += f"\n……（共 {len(result)} 行，只显示前 {max_rows} 行）"
        return text
    if isinstance(result, pd.Series):
        text = result.head(max_rows).to_string()
        if len(result) > max_rows:
            text += f"\n……（共 {len(result)} 行，只显示前 {max_rows} 行）"
        return text
    if isinstance(result, (bool, np.bool_)):
        return "是" if result else "否"
    if isinstance(result, (np.integer, int)):
        return str(int(result))
    if isinstance(result, (np.floating, float)):
        return fmt_num(float(result))
    return str(result)


def repair_code(question, bad_code, error):
    prompt = f"""用户问题：{question}

下面这段 pandas 代码执行失败了：
```python
{bad_code}
```
报错信息：
{error}

请修正代码。要求：
1. 变量 df 是已经加载好的 DataFrame；
2. 只能使用 pandas 和 numpy；
3. 必须把最终结果赋值给变量 result；
4. 不要 print，不要读写文件；
5. 只输出代码本身，不要任何解释，不要用 ``` 包裹。"""
    try:
        return clean_code(chat([
            {"role": "system", "content": "你是 Python 数据分析工程师，只输出可执行的代码。"},
            {"role": "user", "content": prompt},
        ]))
    except Exception:
        return ""


def summarize_answer(question, code, result_text):
    prompt = f"""用户问题：{question}

为了回答这个问题，执行了下面的 pandas 代码：
```python
{code}
```

代码得到的真实结果：
{result_text}

请用简洁、口语化的中文回答用户问题，必须包含具体数字或结论；不要提到代码，不要编造数据。"""
    return chat([
        {"role": "system", "content": "你是数据分析师，回答简洁准确，直接给结论。"},
        {"role": "user", "content": prompt},
    ])


def answer_with_rag(store, df, question, labels=None):
    hits = store.search(question, TOP_K)
    raw = chat(build_rag_messages(df, hits, question, labels), json_mode=True)
    data = parse_json(raw) or {}

    need_code = bool(data.get("need_code"))
    answer = str(data.get("answer") or "").strip()
    code = clean_code(data.get("code"))
    result_text, error = "", ""
    result = None

    if need_code and not code:
        error = "模型说需要计算，但没有给出代码"
    if code:
        for attempt in range(MAX_CODE_RETRY + 1):
            try:
                result, printed = execute_code(code, df)
                result_text = format_result(result) or printed
                error = "" if result_text else "代码执行了，但没有得到 result"
                if result_text:
                    break
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                if attempt < MAX_CODE_RETRY:
                    print(f"       代码报错，正在让模型修正（第 {attempt + 1} 次）...")
                    fixed = repair_code(question, code, error)
                    if not fixed:
                        break
                    code = fixed

    if result_text:
        final = summarize_answer(question, code, result_text)
    elif answer:
        final = answer
    else:
        final = f"抱歉，这个问题没能算出结果。（{error}）"

    return {
        "hits": hits,
        "need_code": need_code,
        "code": code,
        "result_text": result_text,
        "result_obj": result if result_text else None,
        "error": error,
        "answer": final,
    }


def answer_without_rag(question):
    return chat([
        {"role": "system", "content": "你是一个数据分析助手。"},
        {"role": "user", "content": question},
    ])


def auto_analyze(df, data_label, labels=None, progress=None):
    """自动跑完整流程：建知识库 → 设计问题 → 逐题问答/代码 → 返回结果。

    progress(i, n, question, record) 用来在网页上显示进度。
    """
    docs = build_knowledge_base(df, data_label, labels, {})
    store = VectorStore(docs)
    questions = generate_questions(df, labels)
    results = []
    for i, q in enumerate(questions, 1):
        try:
            res = answer_with_rag(store, df, q, labels)
            rec = {
                "question": q,
                "answer": res["answer"],
                "code": res["code"],
                "result": res["result_text"],
                "result_obj": res["result_obj"],
                "hits": [(t, s) for t, s in res["hits"]],
                "error": res["error"],
            }
        except Exception as e:
            rec = {
                "question": q,
                "answer": f"这题出错了：{e}",
                "code": "",
                "result": "",
                "result_obj": None,
                "hits": [],
                "error": str(e),
            }
        results.append(rec)
        if progress is not None:
            progress(i, len(questions), q, rec)
    return docs, store, questions, results


# ==================== 5. 判分 ====================
def numbers_in(text):
    return [float(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*(?:\.\d+)?", str(text))]


def close_enough(a, b):
    if a == b:
        return True
    # 0.4423 和 44.23%（同一个比例的不同写法）也算对上
    for x, y in ((a, b), (a * 100, b), (a, b * 100)):
        if abs(x - y) <= max(0.01, abs(y) * 0.01):
            return True
    return False


def auto_judge(result_text, answer):
    """用“答案里有没有出现真实计算结果”做一个粗略的自动判分，仅供参考。"""
    if not result_text:
        return "需人工"
    truth = numbers_in(result_text)
    if not truth:
        core = result_text.strip().splitlines()[0].strip()
        if core and len(core) <= 20 and core in answer:
            return "✅"
        return "需人工"
    key = truth[0]
    return "✅" if any(close_enough(n, key) for n in numbers_in(answer)) else "❌"


# ==================== 6. 图表 ====================
def save_result_chart(result, question, idx):
    """把某一题的计算结果画成一张图（结果本身适合画图时）。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        title = question if len(question) <= 34 else question[:34] + "…"
        fig = ax = None

        if isinstance(result, pd.Series) and pd.api.types.is_numeric_dtype(result):
            data = result.dropna()
            if not (1 < len(data) <= 20):
                return None
            data = data.head(20)
            fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(data) + 1.5)))
            ax.barh([str(i) for i in data.index][::-1], data.values[::-1], color="#4C78A8")
            ax.set_xlabel("数值")
        elif isinstance(result, pd.DataFrame):
            num = result.select_dtypes(include="number")
            if num.shape[1] < 1 or not (1 < len(result) <= 20):
                return None
            val_col = num.columns[0]
            label_col = next((c for c in result.columns if c not in num.columns), None)
            labels = (result[label_col].astype(str).tolist() if label_col
                      else [str(i) for i in result.index])
            vals = pd.to_numeric(result[val_col], errors="coerce").fillna(0).tolist()
            fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(labels) + 1.5)))
            ax.barh(labels[::-1], vals[::-1], color="#F58518")
            ax.set_xlabel(str(val_col))
        else:
            return None

        ax.set_title(title, fontsize=11)
        fig.tight_layout()
        fp = OUT_DIR / f"图表_问题{idx}.png"
        fig.savefig(fp, dpi=150)
        plt.close(fig)
        return fp
    except Exception:
        return None


def make_charts(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    files = []

    def safe(name):
        return re.sub(r"[^\w\u4e00-\u9fa5-]+", "_", str(name))[:30]

    num_cols = [c for c in df.columns
                if pd.api.types.is_numeric_dtype(df[c])
                and not pd.api.types.is_bool_dtype(df[c])
                and not is_id_like(c, df[c])
                and not is_code_like(c)
                and "type" not in str(c).lower()
                and not any(k in str(c).lower() for k in ("年份", "year"))
                and df[c].nunique(dropna=True) > 5][:2]
    for col in num_cols:
        fig, ax = plt.subplots(figsize=(7, 4))
        df[col].dropna().plot(kind="hist", bins=20, ax=ax, color="#4C78A8", edgecolor="white")
        ax.set_title(f"{col} 的分布")
        ax.set_xlabel(str(col))
        ax.set_ylabel("记录数")
        fig.tight_layout()
        fp = OUT_DIR / f"图表_数值分布_{safe(col)}.png"
        fig.savefig(fp, dpi=150)
        plt.close(fig)
        files.append(fp)

    cat_cols = [c for c in df.columns
                if not pd.api.types.is_numeric_dtype(df[c])
                and not pd.api.types.is_datetime64_any_dtype(df[c])
                and not is_code_like(c)
                and 1 < df[c].nunique(dropna=True) <= 20][:2]
    for col in cat_cols:
        vc = df[col].dropna().astype(str).value_counts().head(10)
        fig, ax = plt.subplots(figsize=(7, 4))
        vc.sort_values().plot(kind="barh", ax=ax, color="#F58518")
        ax.set_title(f"{col} 的类别分布（Top {len(vc)}）")
        ax.set_xlabel("记录数")
        fig.tight_layout()
        fp = OUT_DIR / f"图表_类别分布_{safe(col)}.png"
        fig.savefig(fp, dpi=150)
        plt.close(fig)
        files.append(fp)

    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False).head(15)
    if len(missing):
        fig, ax = plt.subplots(figsize=(7, 4))
        missing.sort_values().plot(kind="barh", ax=ax, color="#E45756")
        ax.set_title("各字段缺失值数量")
        ax.set_xlabel("缺失条数")
        fig.tight_layout()
        fp = OUT_DIR / "图表_缺失值.png"
        fig.savefig(fp, dpi=150)
        plt.close(fig)
        files.append(fp)

    return files


# ==================== 7. 报告 ====================
def md_cell(text, n=90):
    t = str(text).replace("\r", " ").replace("\n", " ").replace("|", "｜").strip()
    return t[:n] + ("…" if len(t) > n else "")


def shorten(text, n=160):
    t = str(text).replace("\r", " ").replace("\n", " ").strip()
    return t[:n] + ("…" if len(t) > n else "")


def build_report(df, data_label, merge_notes, docs, store, records, charts, elapsed, labels=None):
    labels = labels or {}
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rag_ok = sum(r["judge_rag"] == "✅" for r in records)
    no_ok = sum(r["judge_no"] == "✅" for r in records)

    L = []
    L.append("# 任务8：智能数据分析助手（RAG）实验报告\n")
    L.append(f"- 生成时间：{now}")
    L.append(f"- 数据文件：`{data_label}`（合并后 {df.shape[0]} 行 × {df.shape[1]} 列）")
    for n in merge_notes:
        L.append(f"- {n}")
    L.append(f"- 大模型：`{MODEL}`（DeepSeek API）")
    L.append("- 检索方式：字段元信息知识库 + TF-IDF 字符 n-gram 向量化 + 余弦相似度"
             f"（每次取 Top {TOP_K}）")
    L.append(f"- 知识库规模：{len(docs)} 条知识片段")
    L.append(f"- 运行耗时：约 {elapsed:.0f} 秒\n")

    L.append("## 一、运行环境与依赖\n")
    L.append("```")
    L.append(f"Python {sys.version.split()[0]}")
    L.append(f"pandas {pd.__version__}")
    L.append(f"scikit-learn {__import__('sklearn').__version__}")
    L.append(f"openai {__import__('openai').__version__}")
    L.append("```")
    L.append("\n安装命令：`pip install -r requirements.txt`\n")

    L.append("## 二、数据集与知识库\n")
    L.append(f"数据共 {df.shape[0]} 行、{df.shape[1]} 列："
             + "、".join(f"`{c}`" for c in df.columns) + "。\n")
    L.append("每个字段都被整理成一条“字段元信息”文本（类型、缺失、唯一值、"
             "数值统计或高频取值、样例），再加上数据集总览和字段分类，"
             "共同组成检索知识库。\n")
    L.append("| 字段 | 中文名 | 类型 | 非空 | 缺失 | 不同取值 |")
    L.append("| --- | --- | --- | ---: | ---: | ---: |")
    for c in df.columns:
        s = df[c]
        L.append(f"| {md_cell(c, 30)} | {md_cell(labels.get(str(c), ''), 20)} | {s.dtype} | "
                 f"{int(s.notna().sum())} | {int(s.isna().sum())} | {int(s.nunique(dropna=True))} |")
    L.append("")

    L.append("## 三、问题与答案对比\n")
    L.append("| # | 问题 | 无检索（纯大模型） | RAG（检索+计算） | 自动判定 |")
    L.append("| ---: | --- | --- | --- | --- |")
    for i, r in enumerate(records, 1):
        L.append(f"| {i} | {md_cell(r['question'], 40)} | {md_cell(r['no_rag_answer'], 60)} | "
                 f"{md_cell(r['rag_answer'], 60)} | RAG {r['judge_rag']} / 无检索 {r['judge_no']} |")
    L.append("")

    L.append("## 四、逐题详情\n")
    for i, r in enumerate(records, 1):
        L.append(f"### 问题 {i}：{r['question']}\n")
        L.append("**检索到的知识片段：**\n")
        for title, s in r["hits"]:
            L.append(f"- `{s:.3f}` {title}")
        L.append("")
        if r["code"]:
            L.append("**生成的 Pandas 代码：**\n")
            L.append("```python")
            L.append(r["code"])
            L.append("```\n")
        if r["result_text"]:
            L.append("**代码执行结果（真实答案）：**\n")
            L.append("```")
            L.append(r["result_text"])
            L.append("```\n")
        if r.get("chart"):
            L.append(f"![问题{i}的结果图]({r['chart']})\n")
        if r["error"]:
            L.append(f"> 代码执行提示：{r['error']}\n")
        L.append(f"**无检索回答：** {r['no_rag_answer']}\n")
        L.append(f"**RAG 回答：** {r['rag_answer']}\n")
        L.append(f"**自动判定：** RAG {r['judge_rag']}，无检索 {r['judge_no']}"
                 "（自动判定只检查关键数字是否出现，最终请人工看一眼）\n")

    if charts:
        L.append("## 五、图表展示\n")
        for c in charts:
            L.append(f"![{c.stem}]({c.name})\n")

    L.append("## 六、结论\n")
    L.append(
        f"在 {len(records)} 个问题中，RAG 回答有 **{rag_ok}** 个与真实计算结果一致，"
        f"无检索的纯大模型回答有 **{no_ok}** 个一致。\n\n"
        "纯大模型看不到本地数据，只能给出“需要提供数据”之类的泛泛回答，"
        "甚至编造数字；RAG 先从知识库检索字段含义和统计信息，再让模型写 Pandas 代码"
        "在真实数据上计算，因此能给出可核对的具体结果，并且代码可复现、可解释。\n"
    )
    return "\n".join(L)


# ==================== 7. HTML 报告 ====================
HTML_CSS = """
:root{--ink:#1f2329;--muted:#6b7280;--line:#e5e7eb;--blue:#2563eb;--bg:#f8fafc}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:"Microsoft YaHei","PingFang SC",system-ui,sans-serif;line-height:1.7}
.page{max-width:1040px;margin:0 auto;padding:40px 28px 60px}
h1{font-size:30px;margin:0 0 6px}
h2{font-size:21px;margin:38px 0 14px;padding-left:10px;border-left:5px solid var(--blue)}
.sub{color:var(--muted);margin:0 0 6px}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:18px 0}
.stat{background:#fff;border:1px solid var(--line);border-radius:10px;padding:14px}
.stat b{display:block;font-size:22px}
.stat span{color:var(--muted);font-size:13px}
.card{background:#fff;border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin:16px 0;box-shadow:0 1px 2px rgba(0,0,0,.03)}
.q{font-size:18px;font-weight:700;margin:0 0 10px}
.q .no{color:var(--blue)}
.chips{margin:4px 0 10px}
.chip{display:inline-block;background:#eff6ff;color:#1d4ed8;border-radius:999px;padding:2px 10px;font-size:12px;margin:0 6px 6px 0}
pre{background:#0f172a;color:#e2e8f0;border-radius:8px;padding:12px 14px;overflow-x:auto;font-size:13px;line-height:1.55}
pre.result{background:#f1f5f9;color:#0f172a;border:1px solid var(--line)}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:12px}
.answer{border-radius:8px;padding:12px 14px;font-size:14px}
.answer.rag{background:#ecfdf5;border:1px solid #a7f3d0}
.answer.base{background:#f8fafc;border:1px solid var(--line)}
.answer h4{margin:0 0 6px;font-size:13px;color:var(--muted);font-weight:600}
.badge{display:inline-block;border-radius:999px;padding:1px 9px;font-size:12px;font-weight:700}
.ok{background:#d1fae5;color:#065f46}
.bad{background:#fee2e2;color:#991b1b}
.manual{background:#fef3c7;color:#92400e}
table{width:100%;border-collapse:collapse;background:#fff;font-size:13.5px}
th,td{border:1px solid var(--line);padding:8px 10px;text-align:left;vertical-align:top}
th{background:#f1f5f9}
.note{color:var(--muted);font-size:13px}
.chart{text-align:center;margin:12px 0}
.chart img{max-width:100%;border:1px solid var(--line);border-radius:8px}
.meta{color:var(--muted);font-size:13px;margin:2px 0}
@media print{body{background:#fff}.page{padding:0}.card{box-shadow:none;break-inside:avoid}pre{white-space:pre-wrap}h2{break-after:avoid}}
"""


def build_html(df, data_label, merge_notes, docs, records, charts, elapsed, labels=None):
    import base64
    import html as H

    labels = labels or {}

    def esc(x):
        return H.escape(str(x))

    def nl(x):
        return esc(x).replace("\n", "<br>")

    def badge(v):
        cls = "ok" if v == "✅" else ("bad" if v == "❌" else "manual")
        return f'<span class="badge {cls}">{esc(v)}</span>'

    def img_tag(name):
        p = OUT_DIR / name
        if not p.exists():
            return ""
        b64 = base64.b64encode(p.read_bytes()).decode()
        return f'<div class="chart"><img src="data:image/png;base64,{b64}" alt="{esc(name)}"></div>'

    rag_ok = sum(r["judge_rag"] == "✅" for r in records)
    no_ok = sum(r["judge_no"] == "✅" for r in records)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    parts = [f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>任务8 智能数据分析助手（RAG）实验报告</title>
<style>{HTML_CSS}</style></head><body><div class="page">
<h1>任务8：智能数据分析助手（RAG）</h1>
<p class="sub">实验报告 · 生成于 {now}</p>
<p class="meta">姓名：____________　学号：____________　班级：____________</p>

<div class="grid">
  <div class="stat"><b>{df.shape[0]:,}</b><span>合并后记录数</span></div>
  <div class="stat"><b>{df.shape[1]}</b><span>字段数</span></div>
  <div class="stat"><b>{len(records)}</b><span>测试问题数</span></div>
  <div class="stat"><b>{len(docs)}</b><span>知识库片段</span></div>
</div>

<h2>一、数据与知识库</h2>
<div class="card">
<p><b>数据文件：</b>{esc(data_label)}（合并后 {df.shape[0]:,} 行 × {df.shape[1]} 列）</p>
<p><b>大模型：</b>{esc(MODEL)}　<b>检索：</b>字段元信息 + TF-IDF 向量化 + 余弦相似度（Top {TOP_K}）　<b>耗时：</b>约 {elapsed:.0f} 秒</p>
<p class="note">样本说明：</p><ul class="note">{''.join(f'<li>{esc(n)}</li>' for n in merge_notes)}</ul>
</div>"""]

    # 数据概览图
    if charts:
        parts.append('<div class="card"><p><b>数据概览图</b></p>' + "".join(img_tag(c.name) for c in charts) + "</div>")

    # 总表
    rows = []
    for i, r in enumerate(records, 1):
        rows.append(
            f"<tr><td>{i}</td><td>{esc(shorten(r['question'], 46))}</td>"
            f"<td>{esc(shorten(r['no_rag_answer'], 70))}</td>"
            f"<td>{esc(shorten(r['rag_answer'], 70))}</td>"
            f"<td>RAG {badge(r['judge_rag'])}<br>无检索 {badge(r['judge_no'])}</td></tr>"
        )
    parts.append(f"""<h2>二、问题与答案对比</h2>
<table><thead><tr><th>#</th><th>问题</th><th>无检索（纯大模型）</th><th>RAG（检索 + 计算）</th><th>自动判定</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<p class="note">自动判定只检查答案里是否出现真实计算结果中的关键数字，仅供参考，最终以人工核对为准。</p>""")

    # 逐题
    detail = ['<h2>三、逐题详情</h2>']
    for i, r in enumerate(records, 1):
        chips = "".join(f'<span class="chip">{esc(t)} · {s:.2f}</span>' for t, s in r["hits"])
        block = [f'<div class="card"><p class="q"><span class="no">Q{i}.</span> {esc(r["question"])}</p>',
                 f'<div class="chips">{chips}</div>']
        if r["code"]:
            block.append(f'<p class="note">生成的 Pandas 代码</p><pre>{esc(r["code"])}</pre>')
        if r["result_text"]:
            block.append(f'<p class="note">代码执行结果（真实答案）</p><pre class="result">{esc(r["result_text"])}</pre>')
        if r.get("chart"):
            block.append(img_tag(r["chart"]))
        block.append(
            f'<div class="cols">'
            f'<div class="answer base"><h4>无检索（纯大模型）</h4>{nl(r["no_rag_answer"])}</div>'
            f'<div class="answer rag"><h4>RAG（检索 + 代码计算）</h4>{nl(r["rag_answer"])}</div>'
            f'</div><p class="note">判定：RAG {badge(r["judge_rag"])}　无检索 {badge(r["judge_no"])}</p></div>'
        )
        detail.append("".join(block))
    parts.append("".join(detail))

    parts.append(f"""<h2>四、结论</h2>
<div class="card">
<p>在 {len(records)} 个问题中，RAG 回答有 <b>{rag_ok}</b> 个与真实计算结果一致，无检索的纯大模型回答有 <b>{no_ok}</b> 个一致。</p>
<p>纯大模型看不到本地数据，只能给出“需要提供数据”之类的泛泛回答，甚至编造数字；
RAG 先从知识库检索字段含义和统计信息，再让模型写 Pandas 代码在真实数据上计算，
因此能给出可核对的具体结果，并且代码可复现、可解释。</p>
</div>
<p class="note">本报告由 rag_assistant.py 自动生成；图表与代码均来自本次运行。</p>
</div></body></html>""")

    return "".join(parts)


# ==================== 主流程 ====================
def main():
    t0 = time.time()
    line = "=" * 68
    print(line)
    print("任务8：智能数据分析助手（RAG）")
    print(line)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/7] 检查 API Key ...")
    if not API_KEY:
        print("  ✗ 还没有填 API Key")
        print('    打开本文件开头的“基础配置”，把  API_KEY = ""  改成  API_KEY = "sk-..."')
        print("    保存后重新运行即可。")
        sys.exit(1)
    print(f"  ✓ 已读取 Key：{API_KEY[:6]}...{API_KEY[-4:]}")

    print("\n[2/7] 读取数据 ...")
    df, data_label, merge_notes, labels, units = load_dataset()
    print(f"  ✓ {data_label}：{df.shape[0]:,} 行 × {df.shape[1]} 列")
    for note in merge_notes:
        print(f"    · {note}")

    print("\n[3/7] 数据预处理 ...")
    df, before = clean_df(df)
    if df.empty or df.shape[1] == 0:
        raise RuntimeError("清洗后数据为空。请检查文件是否只有表头、或内容是否都在合并单元格里。")
    print(f"  ✓ 字段名已规范化；去掉全空行/列后：{df.shape[0]} 行 × {df.shape[1]} 列"
          f"（原 {before[0]} 行 × {before[1]} 列）")
    show = df.head(3)
    if len(show.columns) > 8:
        show = show.iloc[:, :8]
        print("  （只展示前 8 列）")
    print(show.to_string())

    print("\n[4/7] 构建字段元信息知识库 ...")
    docs = build_knowledge_base(df, data_label, labels, units)
    print(f"  ✓ 共 {len(docs)} 条知识片段")

    print("\n[5/7] 向量化并建立检索索引 ...")
    store = VectorStore(docs)
    print(f"  ✓ 向量维度：{store.matrix.shape[1]}，文档数：{store.matrix.shape[0]}")
    demo_q = "这个数据集有哪些字段？"
    print(f"  检索演示：“{demo_q}”")
    for d, s in store.search(demo_q, 3):
        print(f"      {s:.3f}  {d['title']}")

    print("\n[6/7] 设计问题 ...")
    questions = QUESTIONS or generate_questions(df, labels)
    for i, q in enumerate(questions, 1):
        print(f"  {i}. {q}")

    print("\n[7/7] 逐题问答（RAG vs 无检索）...")
    records = []
    for i, q in enumerate(questions, 1):
        print("\n" + "-" * 68)
        print(f"第 {i}/{len(questions)} 题：{q}")
        t1 = time.time()

        rag = answer_with_rag(store, df, q, labels)
        print("  [检索] 命中的知识片段：")
        for d, s in rag["hits"]:
            print(f"         {s:.3f}  {d['title']}")
        if rag["code"]:
            print("  [RAG] 生成代码：")
            for ln in rag["code"].splitlines():
                print(f"        {ln}")
        if rag["result_text"]:
            print("  [RAG] 计算结果：")
            for ln in rag["result_text"].splitlines()[:12]:
                print(f"        {ln}")
        print(f"  [RAG] 回答：{rag['answer']}")

        no_rag = answer_without_rag(q)
        print(f"  [无检索] 回答：{shorten(no_rag, 160)}")

        judge_rag = auto_judge(rag["result_text"], rag["answer"])
        judge_no = auto_judge(rag["result_text"], no_rag)
        print(f"  [判定] RAG {judge_rag} / 无检索 {judge_no}"
              f"   （用时 {time.time() - t1:.1f} 秒）")

        chart_file = save_result_chart(rag.get("result_obj"), q, i)
        if chart_file:
            print(f"  [图表] 已生成：{chart_file.name}")

        records.append({
            "question": q,
            "hits": [(d["title"], s) for d, s in rag["hits"]],
            "code": rag["code"],
            "result_text": rag["result_text"],
            "error": rag["error"],
            "rag_answer": rag["answer"],
            "no_rag_answer": no_rag,
            "judge_rag": judge_rag,
            "judge_no": judge_no,
            "chart": chart_file.name if chart_file else "",
        })

    print("\n" + line)
    print("生成图表和对比报告 ...")
    charts = []
    try:
        charts = make_charts(df)
        print(f"  ✓ 图表 {len(charts)} 张")
    except Exception as e:
        print(f"  图表生成失败（不影响问答结果）：{e}")

    elapsed = time.time() - t0
    report = build_report(df, data_label, merge_notes, docs, store, records, charts, elapsed, labels)
    report_path = OUT_DIR / "对比报告.md"
    report_path.write_text(report, encoding="utf-8")

    html_path = OUT_DIR / "实验报告.html"
    html_path.write_text(
        build_html(df, data_label, merge_notes, docs, records, charts, elapsed, labels),
        encoding="utf-8",
    )

    raw_path = OUT_DIR / "问答记录.json"
    raw_path.write_text(
        json.dumps(
            [{"question": r["question"],
              "retrieved": [{"title": t, "score": s} for t, s in r["hits"]],
              "code": r["code"],
              "result": r["result_text"],
              "rag_answer": r["rag_answer"],
              "no_rag_answer": r["no_rag_answer"]} for r in records],
            ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + line)
    print("全部完成！")
    print(f"总用时：{elapsed:.0f} 秒")
    print(f"输出目录：{OUT_DIR}")
    print(f"  - 对比报告.md（Markdown 版）")
    print(f"  - 实验报告.html（浏览器打开，Ctrl+P 可存 PDF，推荐交这个）")
    for c in charts:
        print(f"  - {c.name}")
    print(line)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已手动中断。")
    except (FileNotFoundError, RuntimeError) as e:
        # 这些是我们主动抛出的“人话”提示，不用打印吓人的 traceback
        print("\n程序停下了：")
        print(e)
        print("\n按上面的提示处理后，重新运行即可。")
        sys.exit(1)
    except Exception as e:
        print("\n程序运行出错：", e)
        traceback.print_exc()
        print("\n常见原因见《操作手册.md》的“常见报错”一节。")
        sys.exit(1)
