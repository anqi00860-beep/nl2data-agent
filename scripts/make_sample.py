# -*- coding: utf-8 -*-
"""Generate the synthetic sample panel used by the demo (run from repo root)."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(20261106)
rows = []
for i in range(50):
    stkcd = f"{i + 1:06d}"
    name = f"示例公司{i + 1:02d}"
    for year in range(2011, 2021):
        size = rng.normal(21.8, 1.2)
        bank = float(np.clip(0.10 + 0.030 * (size - 21.8) + rng.normal(0, 0.025), 0, 0.60))
        bond = float(np.clip(0.003 * np.exp(size - 22) + rng.normal(0, 0.004), 0, 0.15))
        issue = int(rng.random() < 1 / (1 + np.exp(-(size - 22.2))))
        lev = float(np.clip(rng.normal(0.45, 0.15), 0.05, 0.95))
        rows.append({
            "stkcd": stkcd,
            "年份": year,
            "IssShortName": name,
            "企业规模": round(size, 4),
            "银行融资占比": round(bank, 4),
            "短期银行融资占比": round(bank * 0.7, 4),
            "债券融资占比": round(bond, 4),
            "新发债券占比": round(bond * 0.5, 4),
            "是否发债": issue,
            "债券只数": issue * int(rng.integers(1, 5)),
            "发行总额_亿元": round(bond * 800, 2),
            "加权平均票面利率": round(5.2 - (size - 21.5) + rng.normal(0, 0.3), 2),
            "平均票面利率": round(5.2 - (size - 21.5) + rng.normal(0, 0.3), 2),
            "平均期限": round(float(rng.uniform(1, 10)), 2),
            "资产负债率": round(lev, 4),
            "抵押能力": round(float(np.clip(rng.normal(0.25, 0.10), 0, 0.80)), 4),
            "盈利能力": round(float(rng.normal(0.04, 0.05)), 4),
            "流动性": round(float(np.clip(rng.normal(2.0, 1.0), 0.3, 6.0)), 4),
            "现金占比": round(float(np.clip(rng.normal(0.15, 0.08), 0, 0.60)), 4),
            "存货占比": round(float(np.clip(rng.normal(0.12, 0.08), 0, 0.60)), 4),
            "无形资产占比": round(float(np.clip(rng.normal(0.05, 0.04), 0, 0.40)), 4),
        })

df = pd.DataFrame(rows)
df.to_csv("data/sample_panel.csv", index=False, encoding="utf-8-sig")
print("sample rows:", len(df))
print(df.head(3).to_string(index=False))

