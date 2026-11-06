# -*- coding: utf-8 -*-
"""
智能数据分析 Agent · 网页版
====================================================================
上传一份 Excel/CSV，系统自动完成：
    数据预处理 → 字段元信息构建 → 问题设计 → 大模型问答/代码生成 → 图表展示
然后你可以继续追问。

运行：
    双击 启动网页.bat
    或终端：streamlit run app.py
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import rag_assistant as ra

st.set_page_config(
    page_title="智能数据分析 Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.4rem; max-width: 1180px;}
      h1 {font-size: 1.9rem;}
      [data-testid="stMetricValue"] {font-size: 1.5rem;}
      .hero {background:#f1f5f9;border:1px solid #e2e8f0;border-radius:14px;padding:22px 26px;}
    </style>
    """,
    unsafe_allow_html=True,
)

PANEL_PATH = ra.DATA_DIR / ra.PANEL_FILE
UPLOAD_DIR = ra.DATA_DIR / "uploads"

for k, v in {
    "dataset": None,
    "uid": None,
    "stage": "empty",
    "messages": [],
    "run_now": False,
    "uploader_key": 0,
    "ignore_default": False,
    "compare": False,
    "num_q": 5,
    "pending": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ==================== 读取数据 ====================
def save_upload(uploaded) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / uploaded.name
    path.write_bytes(uploaded.getvalue())
    return path


def load_any(path: Path):
    df, labels, units = ra.load_table(path)
    if df is None or df.empty:
        raise ValueError("这个文件读出来是空表，请检查第一行是不是列名。")
    df, _ = ra.clean_df(df)
    return df, labels, units


@st.cache_data(show_spinner=False)
def load_panel(path: str, mtime: float) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"stkcd": str})
    if "stkcd" in df.columns:
        df["stkcd"] = df["stkcd"].str.zfill(6)
    return df


def set_dataset(name, df, labels, units, source):
    st.session_state.dataset = {
        "name": name,
        "df": df,
        "labels": labels or {},
        "units": units or {},
        "docs": None,
        "store": None,
        "questions": [],
        "results": [],
        "source": source,
    }
    st.session_state.stage = "loaded"
    st.session_state.messages = []
    st.session_state.run_now = False


# 默认面板：没有上传过就先加载它
if st.session_state.dataset is None and not st.session_state.ignore_default and PANEL_PATH.exists():
    _panel_df = load_panel(str(PANEL_PATH), PANEL_PATH.stat().st_mtime)
    set_dataset(PANEL_PATH.name, _panel_df, {}, {}, "default")


# ==================== 工具函数 ====================
def is_num(s):
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def to_chart(obj):
    """把代码执行结果转成能画图的 DataFrame。"""
    if isinstance(obj, pd.Series) and pd.api.types.is_numeric_dtype(obj):
        s = obj.dropna()
        if 1 < len(s) <= 20:
            d = s.to_frame("数值")
            d.index = d.index.astype(str)
            return d
    if isinstance(obj, pd.DataFrame):
        num = obj.select_dtypes(include="number")
        if num.shape[1] >= 1 and 1 < len(obj) <= 20:
            val = num.columns[0]
            label = next((c for c in obj.columns if c not in num.columns), None)
            d = obj[[label, val]].copy() if label else obj[[val]].copy()
            if label:
                d = d.set_index(label)
            d.index = d.index.astype(str)
            return d
    return None


def results_markdown(ds):
    lines = [f"# 自动分析：{ds['name']}", ""]
    for i, r in enumerate(ds["results"], 1):
        lines.append(f"## Q{i}. {r['question']}")
        lines.append(r["answer"])
        if r.get("code"):
            lines += ["", "```python", r["code"], "```"]
        if r.get("result"):
            lines += ["", "```", r["result"], "```"]
        lines.append("")
    return "\n".join(lines)


# ==================== 侧边栏 ====================
with st.sidebar:
    st.title("📊 数据分析 Agent")
    st.caption("上传数据 → 自动分析 → 继续追问")

    api_key = st.text_input("DeepSeek API Key", value=ra.API_KEY, type="password")
    if api_key.strip() != ra.API_KEY:
        ra.API_KEY = api_key.strip()
        ra._client = None
    if ra.API_KEY:
        st.success("API 已连接", icon="✅")
    else:
        st.warning("填 Key 后才能自动分析", icon="⚠️")

    st.divider()
    uploaded = st.file_uploader(
        "上传数据文件",
        type=["xlsx", "xls", "csv"],
        key=f"uploader_{st.session_state.uploader_key}",
        help="支持 Excel / CSV；CSMAR 三行表头会自动识别。一次一个表。",
    )

    auto_run = st.toggle("上传后自动分析", value=True)
    compare = st.toggle("同时显示“无检索”回答", value=False)
    st.session_state.compare = compare
    st.session_state.num_q = st.slider("自动问题数", 3, 6, st.session_state.num_q)

    if st.session_state.dataset is not None:
        st.divider()
        st.markdown(f"**当前数据**：`{st.session_state.dataset['name']}`")
        st.markdown(f"**规模**：{len(st.session_state.dataset['df']):,} 行 × {st.session_state.dataset['df'].shape[1]} 列")
        if st.button("换一份数据", width="stretch"):
            st.session_state.dataset = None
            st.session_state.uid = None
            st.session_state.stage = "empty"
            st.session_state.ignore_default = True
            st.session_state.uploader_key += 1
            st.session_state.messages = []
            st.rerun()

# 处理上传
if uploaded is not None:
    raw = uploaded.getvalue()
    uid = (uploaded.name, len(raw), hashlib.md5(raw).hexdigest())
    if st.session_state.uid != uid:
        with st.spinner("正在读取并清洗数据…"):
            path = save_upload(uploaded)
            df, labels, units = load_any(path)
        set_dataset(uploaded.name, df, labels, units, "upload")
        st.session_state.uid = uid
        st.session_state.stage = "loaded"
        st.session_state.run_now = bool(auto_run and ra.API_KEY)
        st.rerun()


# ==================== 自动分析 ====================
def run_pipeline(ds):
    df = ds["df"]
    log = []
    with st.status("自动分析中…", expanded=True) as status:
        line = f"① 数据预处理：{len(df):,} 行 × {df.shape[1]} 列"
        status.write(line); log.append(line)
        status.write("② 构建字段元信息知识库…")
        docs = ra.build_knowledge_base(df, ds["name"], ds["labels"], ds["units"])
        line = f"② 字段元信息：{len(docs)} 条知识片段"
        status.write(line); log.append(line)

        store = ra.VectorStore(docs)
        line = f"③ 向量索引：{store.matrix.shape[1]} 维"
        status.write(line); log.append(line)

        ra.AUTO_QUESTION_COUNT = st.session_state.num_q
        questions = ra.generate_questions(df, ds["labels"])
        line = "④ 问题设计：" + "；".join(questions)
        status.write(line); log.append(line)

        results = []
        bar = st.progress(0.0, text="逐题分析…")
        for i, q in enumerate(questions, 1):
            status.write(f"⑤ 第 {i}/{len(questions)} 题：{q}")
            try:
                res = ra.answer_with_rag(store, df, q, ds["labels"])
                rec = {
                    "question": q,
                    "answer": res["answer"],
                    "code": res["code"],
                    "result": res["result_text"],
                    "chart": to_chart(res["result_obj"]),
                    "hits": [(t, s) for t, s in res["hits"]],
                    "error": res["error"],
                }
                if st.session_state.compare:
                    rec["no_rag"] = ra.answer_without_rag(q)
            except Exception as e:
                rec = {"question": q, "answer": f"这题出错了：{e}", "code": "",
                       "result": "", "chart": None, "hits": [], "error": str(e)}
            results.append(rec)
            bar.progress(i / len(questions), text=f"已完成 {i}/{len(questions)}")

        ds.update(docs=docs, store=store, questions=questions, results=results, log=log)
        status.update(label="分析完成", state="complete")


def render_results(ds):
    results = ds.get("results") or []
    if not results:
        st.info("还没有分析结果。")
        return
    st.success(f"自动完成 {len(results)} 个问题；下面每个问题都可以展开看代码和计算过程。")
    if ds.get("log"):
        with st.expander("查看分析过程（预处理 → 知识库 → 问题设计 → 逐题）"):
            for line in ds["log"]:
                st.markdown(f"- {line}")
    for i, r in enumerate(results, 1):
        with st.expander(f"Q{i}. {r['question']}", expanded=(i == 1)):
            st.markdown(r["answer"])
            if r.get("code"):
                st.code(r["code"], language="python")
            if r.get("result"):
                st.code(r["result"])
            chart = r.get("chart")
            if chart is not None:
                st.bar_chart(chart)
            if r.get("hits"):
                st.caption("检索片段：" + " · ".join(f"{t}（{s:.2f}）" for t, s in r["hits"][:4]))
            if r.get("no_rag"):
                with st.expander("无检索的纯大模型回答（对比）"):
                    st.markdown(r["no_rag"])
            if r.get("error"):
                st.warning(r["error"])

    st.download_button(
        "下载本次分析（Markdown）",
        results_markdown(ds),
        file_name="自动分析结果.md",
    )


# ==================== 页面 ====================
ds = st.session_state.dataset

if ds is None:
    st.title("上传一份数据，自动完成一次分析")
    st.markdown(
        """
        <div class="hero">
        <b>系统会自动完成：</b><br>
        数据预处理 → 字段元信息构建 → 问题设计 → 大模型问答 / Pandas 代码生成 → 图表展示<br><br>
        然后在「追问」里继续问。支持 Excel / CSV，CSMAR 的三行表头会自动跳过。
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("在左侧「上传数据文件」里选择文件即可。第一次分析大约 1~3 分钟。")
    st.stop()

df = ds["df"]
st.title("智能数据分析 Agent")
st.caption(f"当前数据：{ds['name']}　·　{len(df):,} 行 × {df.shape[1]} 列")

tab1, tab2, tab3 = st.tabs(["📈 数据特征", "🤖 自动分析", "💬 追问"])

# ---------- 数据特征 ----------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("记录数", f"{len(df):,}")
    c2.metric("字段数", f"{df.shape[1]}")
    c3.metric("缺失比例", f"{df.isna().mean().mean() * 100:.1f}%")
    num_all = [c for c in df.columns if is_num(df[c])]
    c4.metric("数值字段", f"{len(num_all)}")

    st.subheader("字段说明")
    rows = []
    for c in df.columns:
        s = df[c]
        rows.append({
            "字段": c,
            "中文名": ds["labels"].get(str(c), ""),
            "类型": str(s.dtype),
            "非空": int(s.notna().sum()),
            "缺失": int(s.isna().sum()),
            "不同取值": int(s.nunique(dropna=True)),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    if num_all:
        st.subheader("数值字段描述统计")
        st.dataframe(df[num_all].describe().T.round(4), width="stretch")

    # 数值分布
    hist_cols = [c for c in num_all
                 if not ra.is_code_like(c) and not ra.is_id_like(c, df[c])
                 and df[c].nunique(dropna=True) > 5][:2]
    for c in hist_cols:
        s = df[c].dropna()
        if len(s) < 2:
            continue
        hist, edges = np.histogram(s, bins=30)
        hdf = pd.DataFrame({"区间中点": (edges[:-1] + edges[1:]) / 2, "记录数": hist}).set_index("区间中点")
        st.caption(f"{c} 的分布")
        st.bar_chart(hdf)

    # 类别分布
    cat_cols = [c for c in df.columns
                if not is_num(df[c])
                and not pd.api.types.is_datetime64_any_dtype(df[c])
                and not ra.is_code_like(c)
                and 1 < df[c].nunique(dropna=True) <= 20][:2]
    for c in cat_cols:
        vc = df[c].dropna().astype(str).value_counts().head(15)
        st.caption(f"{c} 的取值分布")
        st.bar_chart(vc.to_frame("记录数"))

    # 面板专属：规模分组
    if {"企业规模", "银行融资占比"} <= set(df.columns):
        tmp = df.copy()
        tmp["规模分组"] = pd.qcut(
            tmp["企业规模"], 5,
            labels=["最小20%", "20-40%", "40-60%", "60-80%", "最大20%"],
            duplicates="drop",
        )
        cols = [c for c in ["银行融资占比", "债券融资占比", "资产负债率", "是否发债"] if c in tmp.columns]
        grp = tmp.groupby("规模分组", observed=True)[cols].mean().round(4)
        st.subheader("按企业规模五分组")
        st.dataframe(grp, width="stretch")
        if "银行融资占比" in cols:
            st.bar_chart(grp[["银行融资占比"]])

    with st.expander("查看原始数据（前 200 行）"):
        st.dataframe(df.head(200), width="stretch")

# ---------- 自动分析 ----------
with tab2:
    if st.session_state.stage == "done" and ds.get("results"):
        render_results(ds)
    else:
        st.markdown(
            "点击开始后，系统会自动走完：**数据预处理 → 字段元信息 → 问题设计 → 问答/代码 → 图表**。"
        )
        st.caption("会调用 DeepSeek 大约 10~15 次，耗时 1~3 分钟。")
        if st.button("开始自动分析", type="primary", disabled=not ra.API_KEY):
            st.session_state.run_now = True
            st.rerun()
        if not ra.API_KEY:
            st.warning("先在左侧填 DeepSeek API Key。")

    if st.session_state.run_now:
        st.session_state.run_now = False
        try:
            run_pipeline(ds)
            st.session_state.stage = "done"
        except Exception as e:
            st.session_state.stage = "error"
            st.error(f"分析中断：{e}")
        st.rerun()

# ---------- 追问 ----------
with tab3:
    st.markdown("自动分析完成后，可以针对这份数据继续提问。系统仍然会**检索字段 → 写代码 → 本地执行**。")
    if ds.get("questions"):
        st.caption("可以接着问：")
        cols = st.columns(2)
        for i, q in enumerate(ds["questions"][:4]):
            if cols[i % 2].button(q, key=f"follow_{i}", width="stretch"):
                st.session_state.pending = q
                st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("code"):
                with st.expander("生成的 Pandas 代码"):
                    st.code(msg["code"], language="python")
            if msg.get("result"):
                with st.expander("执行结果"):
                    st.code(msg["result"])
            chart = msg.get("chart")
            if chart is not None:
                st.bar_chart(chart)

    if not st.session_state.messages:
        st.info("还没有追问。先在「自动分析」跑一次，或者直接在底部输入问题。")

# ---------- 底部输入 ----------
if "pending" not in st.session_state:
    st.session_state.pending = None

disabled = ds.get("store") is None
prompt = st.chat_input("针对这份数据继续提问…", disabled=disabled)
if prompt:
    st.session_state.pending = prompt

if st.session_state.pending:
    q = st.session_state.pending
    st.session_state.pending = None
    st.session_state.messages.append({"role": "user", "content": q})
    with st.spinner("检索 → 生成代码 → 执行 → 总结…"):
        try:
            res = ra.answer_with_rag(ds["store"], df, q, ds["labels"])
            msg = {
                "role": "assistant",
                "content": res["answer"],
                "code": res["code"],
                "result": res["result_text"],
                "chart": to_chart(res["result_obj"]),
            }
            if st.session_state.compare:
                msg["no_rag"] = ra.answer_without_rag(q)
        except Exception as e:
            msg = {"role": "assistant", "content": f"出错了：{e}"}
    st.session_state.messages.append(msg)
    st.rerun()
