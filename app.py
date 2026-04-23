"""
AI Data Analyst Pro
A Streamlit-based AI Data Analysis Web App.
"""

import io
import os
import warnings
from typing import Optional, List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Data Analyst Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------------------------------------------------------
# Data loading helpers
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame with safe defaults."""
    buffer = io.BytesIO(file_bytes)
    try:
        df = pd.read_csv(buffer)
    except UnicodeDecodeError:
        buffer.seek(0)
        df = pd.read_csv(buffer, encoding="latin-1")
    except Exception:
        buffer.seek(0)
        df = pd.read_csv(buffer, sep=None, engine="python")
    # De-duplicate column names to prevent "label is not unique" errors
    if df.columns.duplicated().any():
        seen: dict = {}
        new_cols = []
        for c in df.columns:
            if c in seen:
                seen[c] += 1
                new_cols.append(f"{c}.{seen[c]}")
            else:
                seen[c] = 0
                new_cols.append(c)
        df.columns = new_cols
    return df


def get_column_types(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    # Try to detect datetime-like object columns
    for c in list(categorical_cols):
        try:
            converted = pd.to_datetime(df[c], errors="raise")
            if converted.notna().sum() > 0:
                datetime_cols.append(c)
                categorical_cols.remove(c)
        except Exception:
            pass
    return numeric_cols, categorical_cols, datetime_cols


# ----------------------------------------------------------------------------
# Profiling
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def generate_profile_html(df: pd.DataFrame, minimal: bool = True) -> Optional[str]:
    """Generate ydata-profiling HTML report. Returns None on failure."""
    try:
        from ydata_profiling import ProfileReport
        profile = ProfileReport(
            df,
            title="AI Data Analyst Pro — Profiling Report",
            explorative=not minimal,
            minimal=minimal,
            progress_bar=False,
        )
        return profile.to_html()
    except Exception as e:
        st.warning(f"Could not generate full profiling report: {e}")
        return None


# ----------------------------------------------------------------------------
# Visualization helpers
# ----------------------------------------------------------------------------
def plot_histogram(df: pd.DataFrame, column: str, bins: int = 30):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df[column].dropna(), bins=bins, kde=True, ax=ax, color="#4C72B0")
    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def plot_bar(df: pd.DataFrame, column: str, top_n: int = 20):
    fig, ax = plt.subplots(figsize=(8, 4))
    counts = df[column].astype(str).value_counts().head(top_n)
    sns.barplot(x=counts.values, y=counts.index, ax=ax, palette="viridis")
    ax.set_title(f"Top {min(top_n, len(counts))} values for {column}")
    ax.set_xlabel("Count")
    ax.set_ylabel(column)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: List[str]):
    if len(numeric_cols) < 2:
        return None
    corr = df[numeric_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(min(1 + 0.6 * len(numeric_cols), 14),
                                    min(1 + 0.5 * len(numeric_cols), 12)))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.75})
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()
    return fig


def plot_line(df: pd.DataFrame, x_col: str, y_cols: List[str]):
    fig, ax = plt.subplots(figsize=(10, 4))
    # Keep only unique columns (avoid duplicate-name issues)
    cols = []
    for c in [x_col] + list(y_cols):
        if c not in cols and c in df.columns:
            cols.append(c)
    plot_df = df.loc[:, cols].copy()
    # If column name is duplicated in source, df.loc returns DataFrame — keep first
    if isinstance(plot_df.get(x_col), pd.DataFrame):
        plot_df = plot_df.loc[:, ~plot_df.columns.duplicated()]
    try:
        plot_df[x_col] = pd.to_datetime(plot_df[x_col], errors="ignore")
    except Exception:
        pass
    plot_df = plot_df.dropna(subset=[x_col])
    try:
        plot_df = plot_df.sort_values(by=x_col)
    except Exception:
        pass
    for y in y_cols:
        if y in plot_df.columns:
            ax.plot(plot_df[x_col], plot_df[y], label=y, linewidth=1.5)
    ax.set_title(f"Trend of {', '.join(y_cols)} over {x_col}")
    ax.set_xlabel(x_col)
    ax.set_ylabel("Value")
    ax.legend(loc="best")
    plt.xticks(rotation=30)
    fig.tight_layout()
    return fig


def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, hue: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue, ax=ax, palette="deep", alpha=0.7)
    ax.set_title(f"{y_col} vs {x_col}")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Smart Insights
# ----------------------------------------------------------------------------
def generate_smart_insights(df: pd.DataFrame) -> List[str]:
    insights: List[str] = []
    rows, cols = df.shape
    insights.append(f"Dataset has **{rows:,} rows** and **{cols} columns**.")

    numeric_cols, categorical_cols, datetime_cols = get_column_types(df)
    insights.append(
        f"Detected **{len(numeric_cols)} numeric**, "
        f"**{len(categorical_cols)} categorical**, "
        f"and **{len(datetime_cols)} datetime** columns."
    )

    # Missing values
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) == 0:
        insights.append("✅ No missing values detected.")
    else:
        top = missing.head(5)
        worst = ", ".join([f"`{c}` ({v:,} / {v / rows:.1%})" for c, v in top.items()])
        insights.append(f"⚠️ Missing values found in **{len(missing)}** columns. Worst: {worst}.")

    # Duplicates
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        insights.append(f"⚠️ Found **{dup_count:,} duplicate rows** ({dup_count / rows:.1%}).")
    else:
        insights.append("✅ No fully duplicated rows.")

    # Top correlations
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(numeric_only=True).abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        pairs = upper.stack().sort_values(ascending=False)
        pairs = pairs[pairs < 0.999]  # exclude self-correlations
        top_pairs = pairs.head(3)
        if len(top_pairs) > 0:
            for (a, b), v in top_pairs.items():
                insights.append(f"🔗 Strong correlation: `{a}` ↔ `{b}` (|r| = {v:.2f}).")

    # High-cardinality categoricals
    for c in categorical_cols:
        nunique = df[c].nunique(dropna=True)
        if nunique > 0 and nunique / max(rows, 1) > 0.5 and rows > 20:
            insights.append(f"🔎 `{c}` has very high cardinality ({nunique:,} unique values) — likely an ID-like column.")
            break

    # Skewed numeric columns
    for c in numeric_cols:
        try:
            s = df[c].dropna()
            if len(s) > 10:
                skew = float(s.skew())
                if abs(skew) > 2:
                    direction = "right" if skew > 0 else "left"
                    insights.append(f"📈 `{c}` is heavily {direction}-skewed (skew = {skew:.2f}).")
                    break
        except Exception:
            continue

    # Datetime range
    for c in datetime_cols:
        try:
            s = pd.to_datetime(df[c], errors="coerce").dropna()
            if len(s) > 0:
                insights.append(f"🗓️ `{c}` ranges from **{s.min().date()}** to **{s.max().date()}**.")
                break
        except Exception:
            continue

    return insights


# ----------------------------------------------------------------------------
# AI Chat (OpenAI code-generation agent — PandasAI-style)
# ----------------------------------------------------------------------------
_CODE_AGENT_SYSTEM = """You are a senior Python data analyst. You will be given a pandas DataFrame named `df`.
Write a single short Python code snippet that answers the user's question.

STRICT RULES:
- Only use pandas (pd), numpy (np), matplotlib.pyplot (plt), and seaborn (sns).
- The DataFrame is already loaded as `df`. Do NOT read files or fetch data.
- Assign the final answer to a variable called `result`.
  - For tabular answers, `result` should be a DataFrame or Series.
  - For numeric/text answers, `result` should be a number or string.
  - For chart answers, set `result` to the matplotlib Figure.
- Do NOT call plt.show().
- Do NOT print anything.
- Do NOT use exec/eval/open/os/sys/subprocess/requests/urllib.
- Return ONLY a fenced ```python ... ``` code block. No prose.
"""


def _safe_exec_pandas_code(code: str, df: pd.DataFrame):
    """Execute LLM-generated pandas code in a restricted namespace."""
    forbidden = ["import os", "import sys", "import subprocess", "open(",
                 "__import__", "eval(", "exec(", "compile(", "input(",
                 "requests", "urllib", "socket", "shutil"]
    lowered = code.lower()
    for token in forbidden:
        if token in lowered:
            raise RuntimeError(f"Generated code blocked (contains `{token}`).")

    fig, _ = plt.subplots(figsize=(8, 4))
    plt.close(fig)  # avoid leaking the placeholder
    safe_globals = {
        "pd": pd, "np": np, "plt": plt, "sns": sns,
        "__builtins__": {
            "len": len, "range": range, "min": min, "max": max, "sum": sum,
            "abs": abs, "round": round, "sorted": sorted, "list": list,
            "dict": dict, "set": set, "tuple": tuple, "str": str, "int": int,
            "float": float, "bool": bool, "enumerate": enumerate, "zip": zip,
            "map": map, "filter": filter, "any": any, "all": all, "print": lambda *a, **k: None,
        },
    }
    local_ns = {"df": df.copy()}
    exec(compile(code, "<llm_code>", "exec"), safe_globals, local_ns)
    return local_ns.get("result", None)


def _extract_code(text: str) -> str:
    if "```" not in text:
        return text.strip()
    parts = text.split("```")
    for chunk in parts:
        chunk = chunk.strip()
        if chunk.lower().startswith("python"):
            return chunk[6:].strip()
    # fallback to first fenced block
    if len(parts) >= 2:
        return parts[1].strip()
    return text.strip()


def _get_secret(name: str) -> str:
    """Read a value from Streamlit secrets first, then environment."""
    try:
        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass
    return os.environ.get(name, "").strip()


def _make_openai_client(api_key: str):
    from openai import OpenAI as OpenAIClient
    # 1) Replit AI Integrations proxy (only inside Replit)
    proxy_base = _get_secret("AI_INTEGRATIONS_OPENAI_BASE_URL")
    proxy_key = _get_secret("AI_INTEGRATIONS_OPENAI_API_KEY")
    if proxy_base and proxy_key:
        return OpenAIClient(base_url=proxy_base, api_key=proxy_key)
    # 2) Custom OpenAI-compatible provider (Groq, OpenRouter, Together, etc.)
    custom_base = _get_secret("OPENAI_BASE_URL")
    if custom_base:
        return OpenAIClient(base_url=custom_base, api_key=api_key)
    # 3) Plain OpenAI
    return OpenAIClient(api_key=api_key)


def run_ai_query(df: pd.DataFrame, question: str, api_key: str, model: str = "gpt-4o-mini"):
    """Ask OpenAI to write pandas code that answers the question, then execute it."""
    try:
        from openai import OpenAI as OpenAIClient  # noqa: F401
    except Exception as e:
        return {"error": f"OpenAI client not available: {e}"}

    try:
        client = _make_openai_client(api_key)
        schema_summary = (
            f"DataFrame `df` shape: {df.shape}\n"
            f"Columns and dtypes:\n{df.dtypes.astype(str).to_dict()}\n"
            f"First 3 rows:\n{df.head(3).to_dict(orient='records')}"
        )
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": _CODE_AGENT_SYSTEM},
                {"role": "user", "content": f"{schema_summary}\n\nQUESTION: {question}"},
            ],
        }
        if not model.startswith(("gpt-5", "o3", "o4")):
            kwargs["temperature"] = 0.1
        resp = client.chat.completions.create(**kwargs)
        raw = resp.choices[0].message.content or ""
        code = _extract_code(raw)
        if not code:
            return {"error": "Model returned no code.", "code": raw}
        try:
            value = _safe_exec_pandas_code(code, df)
        except Exception as exec_err:
            # Try a fallback summary answer
            return {"error": f"Code execution failed: {exec_err}", "code": code}
        return {"result": value, "code": code}
    except Exception as e:
        return {"error": str(e)}


def run_openai_fallback(df: pd.DataFrame, question: str, api_key: str, model: str = "gpt-4o-mini"):
    """Fallback: ask OpenAI directly with a dataset summary as context."""
    try:
        from openai import OpenAI as OpenAIClient
    except Exception as e:
        return {"error": f"OpenAI client not available: {e}"}

    try:
        client = _make_openai_client(api_key)
        summary = {
            "shape": df.shape,
            "columns": df.dtypes.astype(str).to_dict(),
            "head": df.head(5).to_dict(orient="records"),
            "describe": df.describe(include="all").fillna("").to_dict(),
        }
        prompt = (
            "You are a senior data analyst. Given the dataset summary below, "
            "answer the user's question with concise, factual analysis. "
            "If a precise answer requires running code on the full dataset, say so and "
            "provide the best estimate from the summary.\n\n"
            f"DATASET SUMMARY:\n{summary}\n\nQUESTION: {question}"
        )
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if not model.startswith(("gpt-5", "o3", "o4")):
            kwargs["temperature"] = 0.2
        resp = client.chat.completions.create(**kwargs)
        return {"result": resp.choices[0].message.content}
    except Exception as e:
        return {"error": str(e)}


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
def render_sidebar():
    st.sidebar.title("📊 AI Data Analyst Pro")
    st.sidebar.caption("Upload a CSV and explore it with AI.")

    proxy_ready = bool(_get_secret("AI_INTEGRATIONS_OPENAI_BASE_URL")
                       and _get_secret("AI_INTEGRATIONS_OPENAI_API_KEY"))
    env_key = _get_secret("OPENAI_API_KEY")
    custom_base = _get_secret("OPENAI_BASE_URL")

    if proxy_ready:
        st.sidebar.success("🤖 AI ready (Replit AI Integrations)")
        api_key = env_key or "managed-by-replit"
    elif env_key:
        provider = "custom provider" if custom_base else "OpenAI"
        st.sidebar.success(f"🔑 API key loaded ({provider})")
        with st.sidebar.expander("Override key (optional)"):
            override = st.text_input("Use a different key for this session",
                                     type="password", value="")
        api_key = override.strip() or env_key
    else:
        st.sidebar.warning("No API key configured.")
        api_key = st.sidebar.text_input(
            "🔑 API Key (OpenAI / Groq / OpenRouter)",
            type="password",
            value="",
            help="Required for AI Chat. Set OPENAI_API_KEY in Streamlit secrets to skip this.",
        )

    if custom_base:
        st.sidebar.caption(f"Using base URL: `{custom_base}`")

    # Model list — adapt if user pointed to a non-OpenAI provider
    if custom_base and "groq" in custom_base.lower():
        model_options = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]
    elif custom_base and "openrouter" in custom_base.lower():
        model_options = ["openai/gpt-4o-mini", "google/gemini-2.0-flash-exp:free", "meta-llama/llama-3.3-70b-instruct:free"]
    else:
        model_options = ["gpt-5-mini", "gpt-5", "gpt-5.2", "gpt-4.1-mini", "gpt-4o-mini"]

    model = st.sidebar.selectbox("Model", options=model_options, index=0)

    uploaded = st.sidebar.file_uploader("📁 Upload CSV", type=["csv"])

    st.sidebar.markdown("---")
    st.sidebar.subheader("Visualization")
    sample_for_plots = st.sidebar.slider(
        "Sample rows for plots (large datasets)",
        min_value=1000, max_value=200000, value=20000, step=1000,
    )
    bins = st.sidebar.slider("Histogram bins", 10, 100, 30, 5)
    top_n = st.sidebar.slider("Top categories per bar chart", 5, 50, 15, 1)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Profiling")
    enable_profile = st.sidebar.checkbox("Generate profiling report", value=False,
                                         help="Heavy for very large datasets.")
    minimal_profile = st.sidebar.checkbox("Use minimal profile (faster)", value=True)

    return {
        "api_key": api_key.strip(),
        "model": model,
        "uploaded": uploaded,
        "sample_for_plots": sample_for_plots,
        "bins": bins,
        "top_n": top_n,
        "enable_profile": enable_profile,
        "minimal_profile": minimal_profile,
    }


def render_overview(df: pd.DataFrame):
    st.subheader("📄 Dataset Preview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    c4.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")

    with st.expander("Head (first 10 rows)", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)
    with st.expander("Schema & dtypes"):
        schema = pd.DataFrame({
            "column": df.columns,
            "dtype": df.dtypes.astype(str).values,
            "non_null": df.notna().sum().values,
            "nulls": df.isna().sum().values,
            "unique": [df[c].nunique(dropna=True) for c in df.columns],
        })
        st.dataframe(schema, use_container_width=True)
    with st.expander("Describe (numeric + categorical)"):
        try:
            st.dataframe(df.describe(include="all").transpose(), use_container_width=True)
        except Exception as e:
            st.info(f"Could not compute describe(): {e}")


def render_profiling(df: pd.DataFrame, minimal: bool):
    st.subheader("🧪 Data Profiling Report")
    if df.shape[0] > 100_000:
        st.info("Dataset is large — profiling on a 100k random sample for speed.")
        sample = df.sample(100_000, random_state=42)
    else:
        sample = df
    with st.spinner("Generating profiling report..."):
        html = generate_profile_html(sample, minimal=minimal)
    if html:
        st.components.v1.html(html, height=900, scrolling=True)
        st.download_button(
            "⬇️ Download report (HTML)",
            data=html.encode("utf-8"),
            file_name="profiling_report.html",
            mime="text/html",
        )
    else:
        st.warning("Profiling report could not be generated for this dataset.")


def render_visualizations(df: pd.DataFrame, sample_size: int, bins: int, top_n: int):
    st.subheader("📈 Visualizations")
    plot_df = df.sample(sample_size, random_state=42) if len(df) > sample_size else df
    numeric_cols, categorical_cols, datetime_cols = get_column_types(plot_df)

    tabs = st.tabs(["Histogram", "Bar chart", "Correlation heatmap", "Line plot", "Scatter"])

    with tabs[0]:
        if not numeric_cols:
            st.info("No numeric columns to plot.")
        else:
            col = st.selectbox("Numeric column", numeric_cols, key="hist_col")
            st.pyplot(plot_histogram(plot_df, col, bins=bins))

    with tabs[1]:
        if not categorical_cols:
            st.info("No categorical columns to plot.")
        else:
            col = st.selectbox("Categorical column", categorical_cols, key="bar_col")
            st.pyplot(plot_bar(plot_df, col, top_n=top_n))

    with tabs[2]:
        fig = plot_correlation_heatmap(plot_df, numeric_cols)
        if fig is None:
            st.info("Need at least 2 numeric columns for a correlation heatmap.")
        else:
            st.pyplot(fig)

    with tabs[3]:
        x_options = datetime_cols + numeric_cols + categorical_cols
        if not x_options or not numeric_cols:
            st.info("Need a numeric column and an x-axis column for line plots.")
        else:
            x_col = st.selectbox("X axis", x_options, key="line_x")
            y_cols = st.multiselect("Y axis (numeric)", numeric_cols,
                                    default=numeric_cols[:1], key="line_y")
            if y_cols:
                st.pyplot(plot_line(plot_df, x_col, y_cols))

    with tabs[4]:
        if len(numeric_cols) < 2:
            st.info("Need at least 2 numeric columns for a scatter plot.")
        else:
            x = st.selectbox("X", numeric_cols, key="scatter_x")
            y = st.selectbox("Y", [c for c in numeric_cols if c != x], key="scatter_y")
            hue = st.selectbox("Color by (optional)", ["(none)"] + categorical_cols, key="scatter_hue")
            hue_val = None if hue == "(none)" else hue
            st.pyplot(plot_scatter(plot_df, x, y, hue_val))


def render_insights(df: pd.DataFrame):
    st.subheader("💡 Smart Insights")
    insights = generate_smart_insights(df)
    for line in insights:
        st.markdown(f"- {line}")


def render_chat(df: pd.DataFrame, api_key: str, model: str):
    st.subheader("🤖 AI Chat")
    if not api_key:
        st.warning("Add your OpenAI API key in the sidebar to enable AI chat.")
        return

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for entry in st.session_state.chat_history:
        with st.chat_message(entry["role"]):
            content = entry["content"]
            if isinstance(content, pd.DataFrame):
                st.dataframe(content, use_container_width=True)
            else:
                st.markdown(str(content))

    question = st.chat_input("Ask a question about your dataset…")
    if not question:
        return

    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            res = run_ai_query(df, question, api_key=api_key, model=model)
            if "error" in res:
                st.info("Code agent couldn't run — falling back to OpenAI summary mode.")
                fb = run_openai_fallback(df, question, api_key=api_key, model=model)
                if "error" in fb:
                    st.error(f"AI error: {fb['error']}")
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": f"Error: {fb['error']}"}
                    )
                    return
                res = fb

        answer = res["result"]
        if isinstance(answer, pd.DataFrame):
            st.dataframe(answer, use_container_width=True)
        elif isinstance(answer, pd.Series):
            st.dataframe(answer.to_frame(), use_container_width=True)
        elif hasattr(answer, "savefig"):  # matplotlib Figure
            st.pyplot(answer)
        elif isinstance(answer, (int, float, np.integer, np.floating)):
            st.markdown(f"**{answer}**")
        elif isinstance(answer, str):
            st.markdown(answer)
        else:
            st.write(answer)

        if "code" in res:
            with st.expander("View generated code"):
                st.code(res["code"], language="python")

        st.session_state.chat_history.append({"role": "assistant", "content": answer})


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    cfg = render_sidebar()

    st.title("📊 AI Data Analyst Pro")
    st.caption("Upload a CSV → preview, profile, visualize, and chat with your data.")

    if cfg["uploaded"] is None:
        st.info("👈 Upload a CSV file from the sidebar to get started.")
        with st.expander("Don't have a CSV? Try a sample dataset"):
            if st.button("Load Iris sample dataset"):
                try:
                    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
                    st.session_state["sample_df"] = pd.read_csv(url)
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not load sample dataset: {e}")
        if "sample_df" in st.session_state:
            df = st.session_state["sample_df"]
        else:
            return
    else:
        try:
            df = load_csv(cfg["uploaded"].getvalue(), cfg["uploaded"].name)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            return

    # Tabs for the main flow
    tab_overview, tab_profile, tab_viz, tab_insights, tab_chat = st.tabs(
        ["Overview", "Profiling", "Visualizations", "Insights", "AI Chat"]
    )

    with tab_overview:
        render_overview(df)

    with tab_profile:
        if cfg["enable_profile"]:
            render_profiling(df, minimal=cfg["minimal_profile"])
        else:
            st.info("Enable **Generate profiling report** in the sidebar to render the full EDA report.")

    with tab_viz:
        render_visualizations(df, cfg["sample_for_plots"], cfg["bins"], cfg["top_n"])

    with tab_insights:
        render_insights(df)

    with tab_chat:
        render_chat(df, cfg["api_key"], cfg["model"])


if __name__ == "__main__":
    main()
