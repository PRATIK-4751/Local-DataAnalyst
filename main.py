import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import google.generativeai as genai


genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


def ask_gemini_code(columns: list[str], question: str) -> str:
    prompt = f"""
You are a Python data analyst.

You have a pandas DataFrame named df.
The columns are:
{", ".join(columns)}

Rules:
- Do NOT import anything
- Do NOT use print
- Do NOT access files or network
- Use pandas, numpy, matplotlib already available
- Assign outputs ONLY to:
  - result (tables or text)
  - fig (matplotlib figure)
- Return ONLY valid Python code
- No markdown
- No explanations
- No comments

User request:
{question}
"""

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)

    return response.text.strip()


def clean_ai_code(code: str) -> str:
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

    try:
        exec(code, {"__builtins__": safe_builtins}, local_env)
    except Exception as e:
        return {"error": str(e)}

    return {
        "result": local_env.get("result"),
        "fig": local_env.get("fig")
    }


st.set_page_config(page_title="AI Data Analyst", layout="wide")
st.title("AI Data Analyst")
st.caption("Gemini-powered • Chat → Code → Results")

if "chat" not in st.session_state:
    st.session_state.chat = []


uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    columns = df.columns.tolist()

    st.subheader("Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Chat")

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask anything about the data")

    if question:
        st.session_state.chat.append({"role": "user", "content": question})

        with st.spinner("Thinking..."):
            ai_code = ask_gemini_code(columns, question)
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

        st.session_state.chat.append({"role": "assistant", "content": "Done"})

else:
    st.info("Upload a CSV file to begin")
