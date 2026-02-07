import streamlit as st
import pandas as pd
import ollama


def analyze_schema(df: pd.DataFrame) -> pd.DataFrame:
    schema_info = []

    for col in df.columns:
        series = df[col]
        dtype = str(series.dtype)
        missing = series.isna().sum()
        unique = series.nunique()

        role = "categorical"

        if pd.api.types.is_numeric_dtype(series):
            role = "numeric"

        if role == "categorical":
            try:
                parsed = pd.to_datetime(series, errors="coerce")
                if parsed.notna().sum() / len(series) > 0.6:
                    role = "datetime"
                    dtype = "datetime"
            except Exception:
                pass

        schema_info.append({
            "Column": col,
            "Role": role,
            "Data Type": dtype,
            "Missing Values": missing,
            "Unique Values": unique
        })

    return pd.DataFrame(schema_info)


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = []
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            summary.append({
                "Column": col,
                "Mean": series.mean(),
                "Min": series.min(),
                "Max": series.max()
            })
    return pd.DataFrame(summary)


def top_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = []

    for col in df.columns:
        series = df[col]
        if series.isna().all():
            continue

        value_counts = series.value_counts(dropna=True)
        if len(value_counts) == 0:
            continue

        summary.append({
            "Column": col,
            "Top Value": value_counts.index[0],
            "Frequency": value_counts.iloc[0]
        })

    return pd.DataFrame(summary)


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        return pd.DataFrame()
    return numeric_df.corr()


def auto_chart(df: pd.DataFrame, x_col: str, y_col: str):
    if x_col == y_col:
        st.write("X and Y columns must be different.")
        return

    x = df[x_col]
    y = df[y_col]

    if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
        st.line_chart(df[[x_col, y_col]].dropna())
    elif not pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
        grouped = df.groupby(x_col)[y_col].mean()
        st.bar_chart(grouped)
    else:
        st.write("Unsupported chart type")


def handle_questions(df: pd.DataFrame, question: str):
    q = question.lower().strip()

    if q in ["summary", "dataset summary"]:
        return df.describe(include="all")

    if q in ["top", "top rows", "show top"]:
        return df.head(10)

    if q == "correlation":
        return correlation_matrix(df)

    if q.startswith("stats for "):
        col = q.replace("stats for ", "").strip()
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            return df[col].describe()
        else:
            return "Column not found or not numeric"

    return "Question not understood"


def ask_ollama(schema: pd.DataFrame, stats: pd.DataFrame, question: str) -> str:
    prompt = f"""
You are a data analyst.

Dataset schema:
{schema.to_string(index=False)}

Numeric statistics:
{stats.to_string(index=False)}

User question:
{question}

Answer clearly in plain English.
Do not write code.
Do not guess if data is insufficient.
"""

    response = ollama.chat(
        model="minimax-m2:cloud",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


st.set_page_config(page_title="AI ANALYST :)", layout="wide")
st.title("AI ANALYST :)")
st.caption("100% local")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader("Upload a csv file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Your Data")
    st.dataframe(df.head(30), use_container_width=True)

    st.subheader("Data Description")
    col1, col2 = st.columns(2)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])

    schema = analyze_schema(df)
    num_summary = numeric_summary(df)

    st.subheader("Data Types")
    st.dataframe(schema, use_container_width=True)

    st.subheader("Numeric Summary")
    st.dataframe(num_summary, use_container_width=True)

    st.subheader("Top Values")
    st.dataframe(top_value_summary(df), use_container_width=True)

    st.subheader("Correlation Matrix")
    corr = correlation_matrix(df)
    if corr.empty:
        st.write("Not enough numeric columns")
    else:
        st.dataframe(corr, use_container_width=True)

    st.subheader("Auto Chart")
    cols = df.columns.tolist()
    x_col = st.selectbox("X column", cols)
    y_col = st.selectbox("Y column", cols)
    auto_chart(df, x_col, y_col)

    chat_mode = st.radio(
        "Chat mode",
        ["Smart (Rules + AI)", "Plain Chat (AI only)"],
        horizontal=True
    )

    st.subheader("Chat with your Data")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask anything about the data...")

    if question:
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        if chat_mode == "Plain Chat (AI only)":
            with st.spinner("Thinking..."):
                result = ask_ollama(schema, num_summary, question)
            source = "AI"
        else:
            result = handle_questions(df, question)
            source = "RULE"

            if isinstance(result, str) and result == "Question not understood":
                with st.spinner("Thinking..."):
                    result = ask_ollama(schema, num_summary, question)
                source = "AI"

        if isinstance(result, pd.DataFrame):
            answer = f"**[{source}]**\n\n{result.to_markdown()}"
        else:
            answer = f"**[{source}]** {result}"

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer
        })

        with st.chat_message("assistant"):
            st.markdown(answer)

else:
    st.write("Please upload a file")
