import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import io
import contextlib


# Read Ollama Cloud API key from environment
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")



# Analyze dataset schema so AI understands columns
def analyze_schema(df: pd.DataFrame) -> pd.DataFrame:
    schema = []

    for col in df.columns:
        series = df[col]
        role = "categorical"

        if pd.api.types.is_numeric_dtype(series):
            role = "numeric"
        else:
            try:
                parsed = pd.to_datetime(series, errors="coerce")
                if parsed.notna().sum() / len(series) > 0.6:
                    role = "datetime"
            except Exception:
                pass

        schema.append({
            "Column": col,
            "Role": role,
            "Missing": series.isna().sum(),
            "Unique": series.nunique()
        })

    return pd.DataFrame(schema)


# Ask Ollama Cloud to generate Python analysis code
OLLAMA_API_URL = "https://api.ollama.com/v1/chat/completions"

def ask_ollama_code(schema: pd.DataFrame, question: str) -> str:
    headers = {
        "Authorization": f"Bearer {OLLAMA_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""
You are a Python data analyst.

You are given a pandas DataFrame called df.

Available objects:
- df (pandas DataFrame)
- pd (pandas)
- np (numpy)
- plt (matplotlib.pyplot)

Rules:
- DO NOT import anything
- DO NOT access files, OS, or network
- Assign outputs only to:
    - result
    - fig
- Return ONLY valid Python code
- No markdown
- No explanations

Dataset schema:
{schema.to_string(index=False)}

User request:
{question}
"""

    payload = {
        "model": "qwen3-coder:480b-cloud",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }

    response = requests.post(
        OLLAMA_API_URL,
        headers=headers,
        json=payload,
        timeout=60
    )

    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]


# Clean AI output to remove imports or markdown
def clean_ai_code(code: str) -> str:
    code = code.strip()

    if code.startswith("```"):
        code = code.split("```")[1]

    cleaned = []
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("import"):
            continue
        if stripped.startswith("from "):
            continue
        if "__import__" in stripped:
            continue
        cleaned.append(line)

    return "\n".join(cleaned).strip()


# Execute AI-generated code in a sandbox
def execute_ai_code(df: pd.DataFrame, code: str):
    safe_builtins = {
        "dict": dict,
        "list": list,
        "set": set,
        "tuple": tuple,
        "len": len,
        "range": range,
        "min": min,
        "max": max,
        "sum": sum,
        "abs": abs
    }

    local_env = {
        "df": df.copy(),
        "pd": pd,
        "np": np,
        "plt": plt,
        "result": None,
        "fig": None
    }

    stdout = io.StringIO()

    try:
        with contextlib.redirect_stdout(stdout):
            exec(code, {"__builtins__": safe_builtins}, local_env)
    except Exception as e:
        return {"error": str(e)}

    return {
        "result": local_env.get("result"),
        "fig": local_env.get("fig")
    }


# Streamlit UI configuration
st.set_page_config(
    page_title="AI Data Analyst",
    layout="wide"
)

st.title("AI Data Analyst")
st.caption("Cloud AI • ChatGPT-style • CSV → Analysis → Charts")

# Initialize chat memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# CSV uploader
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    schema = analyze_schema(df)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Dataset Schema")
    st.dataframe(schema, use_container_width=True)

    st.subheader("Chat")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask anything about the data")

    if question:
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        with st.spinner("AI is analyzing and executing..."):
            ai_code = ask_ollama_code(schema, question)
            clean_code = clean_ai_code(ai_code)
            output = execute_ai_code(df, clean_code)

        with st.chat_message("assistant"):
            if "error" in output:
                st.error(output["error"])
            else:
                if output["fig"] is not None:
                    st.pyplot(output["fig"])
                if isinstance(output["result"], pd.DataFrame):
                    st.dataframe(output["result"], use_container_width=True)
                elif output["result"] is not None:
                    st.write(output["result"])

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": "Analysis completed"
        })

else:
    st.info("Upload a CSV file to begin")
