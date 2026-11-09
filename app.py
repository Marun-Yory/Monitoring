
import os
from dataclasses import asdict

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from assistant import create_assistant
from db_save import save_conversation
from db_feedback import save_feedback
from judge import evaluate_relevance
from db_query import get_conversations, get_stats

load_dotenv()

st.set_page_config(
    page_title="Currency Assistant",
    page_icon="💱",
    layout="wide",
)

DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.7-max-2026-06-08"

MODEL_PRESETS = {
    "qwen3.7-max-2026-06-08": (0.96, 3.84),
    "qwen-plus": (0.40, 1.20),
    "qwen-turbo": (0.05, 0.20),
    "qwen-max": (1.60, 6.40),
}

for key, value in {
    "api_key": os.getenv("OPENAI_API_KEY", ""),
    "model_name": os.getenv("LLM_MODEL", DEFAULT_MODEL),
    "base_url": os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL),
    "input_token_price": float(os.getenv("INPUT_TOKEN_PRICE", 0.96)),
    "output_token_price": float(os.getenv("OUTPUT_TOKEN_PRICE", 3.84)),
}.items():
    if key not in st.session_state:
        st.session_state[key] = value


@st.cache_resource
def get_assistant(api_key, model, base_url, input_price, output_price):
    return create_assistant(
        api_key=api_key,
        model=model,
        base_url=base_url,
        input_price_per_million=input_price,
        output_price_per_million=output_price,
    )


def chat_page():
    st.title("💱 Currency Assistant")

    col1, col2 = st.columns(2)
    with col1:
        st.caption(f"Model: {st.session_state.model_name}")
    with col2:
        if st.button("📊 Dashboard"):
            st.switch_page(page_dashboard)

    if not st.session_state.api_key:
        st.warning("Configure your API key in Settings.")
        if st.button("⚙️ Settings"):
            st.switch_page(page_settings)
        return

    question = st.text_input(
        "Ask a currency question:",
        placeholder="What currency does Japan use?",
    )

    if st.button("🚀 Ask", type="primary"):
        if not question.strip():
            st.warning("Enter a question.")
            return

        with st.spinner("Thinking..."):
            assistant = get_assistant(
                st.session_state.api_key,
                st.session_state.model_name,
                st.session_state.base_url,
                st.session_state.input_token_price,
                st.session_state.output_token_price,
            )

            answer = assistant.rag(question)
            record = assistant.last_call

            conversation_id = save_conversation(record, question)

            relevance, explanation = evaluate_relevance(
                question,
                answer,
                model=st.session_state.model_name,
                api_key=st.session_state.api_key,
                base_url=st.session_state.base_url,
            )

            save_feedback(
                conversation_id,
                "judge",
                relevance=relevance,
                explanation=explanation,
            )

            st.session_state.last_qa = {
                "question": question,
                "answer": answer,
                "record": record,
                "relevance": relevance,
                "explanation": explanation,
                "id": conversation_id,
            }
            st.session_state.user_feedback_submitted = None

    qa = st.session_state.get("last_qa")

    if qa:
        st.divider()
        st.chat_message("user").write(qa["question"])

        with st.chat_message("assistant"):
            st.write(qa["answer"])

            rec = qa["record"]
            cols = st.columns(4)
            cols[0].metric("Response Time", f"{rec.response_time:.2f}s")
            cols[1].metric("Prompt Tokens", rec.prompt_tokens)
            cols[2].metric("Completion Tokens", rec.completion_tokens)
            cols[3].metric("Cost", f"${rec.cost:.6f}")

            with st.expander(f"Judge: {qa['relevance']}"):
                st.write(qa["explanation"])

            col1, col2 = st.columns(2)

            for col, label, score in [
                (col1, "👍 Helpful", 1),
                (col2, "👎 Not Helpful", -1),
            ]:
                if col.button(label, use_container_width=True):
                    save_feedback(qa["id"], "user", score=score)
                    st.session_state.user_feedback_submitted = score

            if st.session_state.user_feedback_submitted == 1:
                st.success("Thank you for your feedback!")
            elif st.session_state.user_feedback_submitted == -1:
                st.info("Thank you. We'll improve our answers.")


def dashboard_page():
    st.title("📊 Currency Dashboard")

    if st.button("💬 Back to Chat"):
        st.switch_page(page_chat)

    stats = get_stats()
    cols = st.columns(4)

    cols[0].metric("Conversations", stats.total)
    cols[1].metric("Avg Response", f"{stats.avg_response_time or 0:.2f}s")
    cols[2].metric("Total Cost", f"${stats.total_cost or 0:.4f}")
    cols[3].metric("Avg Tokens", f"{stats.avg_tokens or 0:.0f}")

    records = get_conversations(limit=100)
    df = pd.DataFrame([asdict(r) for r in records])

    if not df.empty:
        st.subheader("Cost Over Time")
        st.line_chart(df, x="timestamp", y="cost")

        st.subheader("Response Time")
        st.line_chart(df, x="timestamp", y="response_time")

    st.subheader("Recent Conversations")

    for record in records[:20]:
        st.write(f"**{record.prompt[:80]}**")
        st.write(record.answer[:200])
        st.caption(
            f"{record.response_time:.2f}s | ${record.cost:.6f}"
        )
        st.divider()


def settings_page():
    st.title("⚙️ Settings")

    if st.button("💬 Back to Chat"):
        st.switch_page(page_chat)

    with st.form("settings"):
        api_key = st.text_input(
            "API Key",
            value=st.session_state.api_key,
            type="password",
        )
        model = st.text_input(
            "Model",
            value=st.session_state.model_name,
        )
        base_url = st.text_input(
            "Base URL",
            value=st.session_state.base_url,
        )

        input_price = st.number_input(
            "Input Price / 1M Tokens",
            min_value=0.0,
            value=st.session_state.input_token_price,
        )
        output_price = st.number_input(
            "Output Price / 1M Tokens",
            min_value=0.0,
            value=st.session_state.output_token_price,
        )

        if st.form_submit_button("Save Settings"):
            st.session_state.api_key = api_key.strip()
            st.session_state.model_name = model.strip()
            st.session_state.base_url = base_url.strip()
            st.session_state.input_token_price = input_price
            st.session_state.output_token_price = output_price

            st.cache_resource.clear()
            st.success("Settings saved!")

    st.subheader("Model Presets")

    cols = st.columns(4)

    for col, (name, prices) in zip(cols, MODEL_PRESETS.items()):
        if col.button(name, use_container_width=True):
            st.session_state.model_name = name
            st.session_state.input_token_price = prices[0]
            st.session_state.output_token_price = prices[1]
            st.cache_resource.clear()
            st.rerun()


page_chat = st.Page(chat_page, title="Chat", icon="💱", default=True)
page_dashboard = st.Page(dashboard_page, title="Dashboard", icon="📊")
page_settings = st.Page(settings_page, title="Settings", icon="⚙️")

st.navigation([page_chat, page_dashboard, page_settings]).run()
