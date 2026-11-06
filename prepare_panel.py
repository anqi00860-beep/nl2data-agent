# -*- coding: utf-8 -*-
"""
把 bond.xlsx（债券）和 company.xlsx（公司财务）处理成“公司-年度”分析面板。

处理步骤：
    1. 读两张原始表（自动跳过 CSMAR 的中文名行、单位行）
    2. 债券表按 (股票代码, 发行年份) 聚合成公司-年度指标
    3. 与公司财务面板左连接（没发债的公司也保留，是否发债=0）
    4. 生成论文常用的变量：企业规模、银行融资占比、债券融资占比、资产负债率……
    5. 按论文口径筛样本、1%/99% 缩尾
    6. 输出 data/公司年度面板.csv + outputs/数据处理说明.md + 描述统计/分组表/图

运行：右键本文件 → 运行 'prepare_panel'
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

import rag_assistant as ra  # 复用它的读表、清洗、找列函数

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def log(msg):
    print(msg, flush=True)


def load_raw(filename):
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"找不到 {path}")
    df, labels, units = ra.load_table(path)
    df, _ = ra.clean_df(df)
    return df, labels, units


def prepare_key_and_year(df, name):
    aliases = (ra.MERGE_KEY,) + tuple(ra.MERGE_KEY_ALIASES)
    col = ra.find_column(df, aliases)
    if col is None:
        raise RuntimeError(f"{name} 里找不到股票代码列，现有列：{list(df.columns)}")
    if col != "stkcd":
        df = df.rename(columns={col: "stkcd"})
    df["stkcd"] = ra.normalize_key(df["stkcd"])
    date_col = ra.find_date_column(df)
    if date_col is None:
        raise RuntimeError(f"{name} 里找不到日期列，无法按年份对齐")
    df["年份"] = df[date_col].dt.year
    df = df.dropna(subset=["stkcd", "年份"])
    df["年份"] = df["年份"].astype(int)
    log(f"  {name}：{len(df):,} 行，代码列 {col}，日期列 {date_col}")
    return df


def weighted_rate(sub):
    amt = pd.to_numeric(sub["Acisuquty"], errors="coerce")
    rate = pd.to_numeric(sub["Intrrate"], errors="coerce")
    m = amt.notna() & rate.notna() & (amt > 0)
    if not m.any():
        return np.nan
    return float((amt[m] * rate[m]).sum() / amt[m].sum())


def main():
    log("=" * 68)
    log("生成“公司-年度”分析面板")
    log("=" * 68)

    log("\n[1/6] 读取原始数据 ...")
    bond, b_labels, _ = load_raw("bond.xlsx")
    company, c_labels, _ = load_raw("company.xlsx")
    bond = prepare_key_and_year(bond, "bond")
    company = prepare_key_and_year(company, "company")

    log("\n[2/6] 债券表聚合成 公司-年度 ...")
    bond["Acisuquty"] = pd.to_numeric(bond["Acisuquty"], errors="coerce")
    bond["Intrrate"] = pd.to_numeric(bond["Intrrate"], errors="coerce")
    bond["Term"] = pd.to_numeric(bond["Term"], errors="coerce")

    agg = (
        bond.groupby(["stkcd", "年份"], as_index=False)
        .agg(
            债券只数=("Liscd", "count"),
            发行总额_亿元=("Acisuquty", "sum"),
            平均票面利率=("Intrrate", "mean"),
            平均期限=("Term", "mean"),
        )
    )
    wr = (
        bond.groupby(["stkcd", "年份"])
        .apply(weighted_rate, include_groups=False)
        .rename("加权平均票面利率")
        .reset_index()
    )
    agg = agg.merge(wr, on=["stkcd", "年份"], how="left")
    log(f"  聚合后 {len(agg):,} 个 (股票代码, 年份) 组合")

    log("\n[3/6] 合并公司财务面板 ...")
    company = company.drop_duplicates(subset=["stkcd", "年份"], keep="last")
    panel = company.merge(agg, on=["stkcd", "年份"], how="left")
    panel["债券只数"] = panel["债券只数"].fillna(0).astype(int)
    panel["发行总额_亿元"] = panel["发行总额_亿元"].fillna(0.0)
    panel["是否发债"] = (panel["债券只数"] > 0).astype(int)
    log(f"  公司面板 {len(company):,} 行 + 债券聚合 {len(agg):,} 行 → {len(panel):,} 行")

    log("\n[4/6] 生成分析变量 ...")
    def num(col):
        return pd.to_numeric(panel[col], errors="coerce")

    assets = num("A100000")
    panel["企业规模"] = np.log(assets.where(assets > 0))
    panel["银行融资占比"] = (num("A210101").fillna(0) + num("A220101").fillna(0)) / assets
    panel["短期银行融资占比"] = num("A210101").fillna(0) / assets
    panel["债券融资占比"] = num("A220201").fillna(0) / assets
    panel["新发债券占比"] = panel["发行总额_亿元"] * 1e8 / assets
    panel["资产负债率"] = num("A200000") / assets
    panel["抵押能力"] = num("FixedAssets") / assets
    panel["盈利能力"] = num("T40401")
    panel["流动性"] = num("T10100")
    panel["现金占比"] = num("Cash") / assets
    panel["存货占比"] = num("A111201") / assets
    panel["无形资产占比"] = num("IntangibleAssets") / assets

    log("\n[5/6] 按论文口径筛选样本 + 缩尾 ...")
    n0 = len(panel)
    panel = panel[assets > 0]
    n1 = len(panel)
    panel = panel[panel["资产负债率"].between(0, 1)]
    n2 = len(panel)
    panel = panel[panel["企业规模"].notna()]
    n3 = len(panel)

    ratio_cols = [
        "银行融资占比", "短期银行融资占比", "债券融资占比", "新发债券占比",
        "资产负债率", "抵押能力", "盈利能力", "流动性",
        "现金占比", "存货占比", "无形资产占比",
    ]
    for col in ratio_cols:
        s = panel[col]
        lo, hi = s.quantile(0.01), s.quantile(0.99)
        panel[col] = s.clip(lo, hi)

    log(f"  原始 {n0:,} 行 → 总资产>0 {n1:,} → 资产负债率 0~1 {n2:,} → 关键变量非空 {n3:,}")
    log(f"  对 {len(ratio_cols)} 个连续变量做 1%/99% 缩尾")

    keep = [
        "stkcd", "年份", "IssShortName",
        "企业规模", "银行融资占比", "短期银行融资占比", "债券融资占比", "新发债券占比",
        "是否发债", "债券只数", "发行总额_亿元", "加权平均票面利率", "平均票面利率", "平均期限",
        "资产负债率", "抵押能力", "盈利能力", "流动性", "现金占比", "存货占比", "无形资产占比",
        "A100000", "A200000", "A210101", "A220101", "A220201",
        "FixedAssets", "Cash", "A111201", "IntangibleAssets",
    ]
    keep = [c for c in keep if c in panel.columns]
    panel = panel[keep].sort_values(["stkcd", "年份"]).reset_index(drop=True)

    panel_path = DATA_DIR / "公司年度面板.csv"
    panel.to_csv(panel_path, index=False, encoding="utf-8-sig")
    log(f"  ✓ 面板已保存：{panel_path}（{len(panel):,} 行 × {len(panel.columns)} 列）")

    # ---------- 描述统计 ----------
    desc_vars = [
        "企业规模", "银行融资占比", "短期银行融资占比", "债券融资占比", "新发债券占比",
        "资产负债率", "抵押能力", "盈利能力", "流动性", "是否发债", "债券只数", "加权平均票面利率",
    ]
    desc = panel[desc_vars].describe().T[["count", "mean", "std", "min", "50%", "max"]]
    desc.columns = ["样本量", "均值", "标准差", "最小值", "中位数", "最大值"]
    desc = desc.round(4)
    desc.to_csv(OUT_DIR / "描述统计.csv", encoding="utf-8-sig")
    log("\n描述统计：")
    print(desc.to_string())

    # ---------- 规模分组 ----------
    panel["规模分组"] = pd.qcut(
        panel["企业规模"], 5,
        labels=["最小20%", "20-40%", "40-60%", "60-80%", "最大20%"],
        duplicates="drop",
    )
    group_cols = ["银行融资占比", "债券融资占比", "资产负债率", "抵押能力", "盈利能力", "是否发债", "债券只数"]
    grp = panel.groupby("规模分组", observed=True)[group_cols].mean().round(4)
    grp.to_csv(OUT_DIR / "规模分组.csv", encoding="utf-8-sig")
    log("\n按企业规模五分组（组内均值）：")
    print(grp.to_string())

    # 小企业 vs 大企业
    panel["规模类型"] = np.where(
        panel["企业规模"] > panel["企业规模"].median(), "大企业", "小企业"
    )
    split = panel.groupby("规模类型", observed=True)[group_cols].mean().round(4)
    split.to_csv(OUT_DIR / "大小企业对比.csv", encoding="utf-8-sig")
    log("\n小企业 vs 大企业（均值）：")
    print(split.to_string())

    # ---------- 图 ----------
    fig, ax = plt.subplots(figsize=(7.5, 4))
    grp["银行融资占比"].plot(kind="bar", ax=ax, color="#4C78A8", rot=0)
    ax.set_title("按企业规模分组：银行融资占总资产比重")
    ax.set_ylabel("银行融资占比")
    ax.set_xlabel("企业规模分组")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "图_规模分组_银行融资.png", dpi=150)
    plt.close(fig)

    yearly = panel.groupby("年份").agg(
        发债企业数=("是否发债", "sum"),
        平均债券融资占比=("债券融资占比", "mean"),
        平均银行融资占比=("银行融资占比", "mean"),
    )
    yearly.to_csv(OUT_DIR / "年度趋势.csv", encoding="utf-8-sig")
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.bar(yearly.index, yearly["发债企业数"], color="#F58518", alpha=0.75, label="发债企业数")
    ax1.set_ylabel("发债企业数")
    ax1.set_xlabel("年份")
    ax2 = ax1.twinx()
    ax2.plot(yearly.index, yearly["平均债券融资占比"], color="#4C78A8", marker="o", label="平均债券融资占比")
    ax2.set_ylabel("平均债券融资占比")
    fig.suptitle("发债企业数量与平均债券融资占比（年度）")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "图_年度趋势.png", dpi=150)
    plt.close(fig)

    # ---------- 处理说明 ----------
    notes = f"""# 数据处理说明

## 1. 数据来源
- `bond.xlsx`：债券发行明细，{len(bond):,} 行。股票代码列 `IssSymbol`，日期列 `Ipodt`。
- `company.xlsx`：公司财务面板，{len(company):,} 行。股票代码列 `Stkcd`，日期列 `Accper`。

## 2. 处理步骤
1. 跳过 CSMAR 的中文名行和单位行，去掉 `CSR_Finidx.` 等表名前缀。
2. 债券表按 **(股票代码, 发行年份)** 聚合：
   - 债券只数、发行总额（亿元）、平均票面利率、加权平均票面利率、平均期限。
3. 与公司财务面板按 (股票代码, 年份) 左连接，没发债的公司保留，`是否发债=0`。
4. 生成变量（对应论文表 1 的思路）：

| 变量 | 计算 | 对应论文 |
| --- | --- | --- |
| 企业规模 | ln(总资产) | SIZE1 |
| 银行融资占比 | (短期借款 + 长期借款) / 总资产 | STLTLOAN |
| 短期银行融资占比 | 短期借款 / 总资产 | STLOAN |
| 债券融资占比 | 应付债券 / 总资产 | 本文扩展（债券 vs 银行） |
| 新发债券占比 | 当年发行总额(元) / 总资产 | 本文扩展 |
| 资产负债率 | 负债合计 / 总资产 | 资本结构控制变量 |
| 抵押能力 | 固定资产 / 总资产 | TANGI1 |
| 盈利能力 | 资产收益率 A | PROFIT3 |
| 流动性 | 流动比率 | 控制变量 |
| 现金占比、存货占比、无形资产占比 | 对应科目 / 总资产 | 控制变量 |

> 借款、应付债券等科目为空时按 0 处理（视为没有该项融资），这是文献里的常见做法。

5. 样本筛选（参照论文）：
   - 总资产 > 0；
   - 资产负债率在 0~1 之间；
   - 关键变量非空；
   - 连续变量 1%/99% 缩尾。

## 3. 样本变化
原始 {n0:,} 行 → 总资产>0 后 {n1:,} 行 → 资产负债率 0~1 后 {n2:,} 行 → 关键变量非空 {n3:,} 行。

## 4. 与论文的差异（写进报告，避免被问）
- 论文用 1991–2009 年上市公司，本文用 {panel['年份'].min()}–{panel['年份'].max()} 年发债相关样本；
- 论文的“股权融资”需要 IPO/增发数据，本数据没有，因此把对照口径改为**债券融资 vs 银行融资**；
- 论文控制变量中的营业收入、净利润、应收账款、托宾 Q 等本数据没有，只能用现有财务科目替代；
- 公司表不含银行等金融机构，样本以非金融上市公司为主。

## 5. 输出文件
- `data/公司年度面板.csv`：分析面板（{len(panel):,} 行 × {len(panel.columns)} 列）
- `outputs/描述统计.csv`
- `outputs/规模分组.csv`
- `outputs/大小企业对比.csv`
- `outputs/年度趋势.csv`
- `outputs/图_规模分组_银行融资.png`
- `outputs/图_年度趋势.png`
"""
    (OUT_DIR / "数据处理说明.md").write_text(notes, encoding="utf-8")
    log(f"\n✓ 处理说明：{OUT_DIR / '数据处理说明.md'}")
    log("=" * 68)
    log("完成。下一步：直接运行 rag_assistant.py（它会优先读取这个面板）")


if __name__ == "__main__":
    main()
