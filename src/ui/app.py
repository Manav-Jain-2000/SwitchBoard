import json
import uuid
import os

import requests
import streamlit as st

# Configuration
API_URL = os.getenv("SWITCHBOARD_API_URL", "http://localhost:8000").rstrip("/")
BASELINE_LABEL = "GPT-5"

st.set_page_config(page_title="SwitchBoard Agent", layout="wide")


def init_state() -> None:
    defaults = {
        "session_id": str(uuid.uuid4()),
        "messages": [],
        "current_trace": [],
        "knn_scores": {},
        "waiting_for_clarification": False,
        "total_saved": 0.0,
        "total_cost": 0.0,
        "total_latency": 0.0,
        "last_routed_models": [],
        "last_used_models": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def parse_sse_stream(response):
    final_payload = None
    interrupt_question = ""

    for line in response.iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8")
        if not decoded.startswith("data: ") or decoded == "data: [DONE]":
            continue

        data = json.loads(decoded[6:])
        event_type = data.get("type")
        if event_type == "trace":
            st.session_state.current_trace.append(data["entry"])
            if data.get("knn_scores"):
                st.session_state.knn_scores = data["knn_scores"]
        elif event_type == "interrupt":
            st.session_state.waiting_for_clarification = True
            interrupt_question = data.get("question", "")
        elif event_type == "final":
            final_payload = data

    return final_payload, interrupt_question


def render_final_message(final_payload: dict) -> str:
    """Returns ONLY the answer text for the chat bubble.
    Cost / latency / routed / used are stored in session_state for the sidebar widgets;
    they are intentionally NOT appended to the user-facing answer.
    """
    response_text = final_payload.get("response", "")
    cost = float(final_payload.get("total_cost", 0.0))
    latency = float(final_payload.get("total_latency", 0.0))
    saved = float(final_payload.get("cost_saved", 0.0))
    routed = final_payload.get("routed_models", [])
    used = final_payload.get("used_models", [])

    st.session_state.total_cost += cost
    st.session_state.total_saved += saved
    st.session_state.total_latency += latency
    st.session_state.last_routed_models = routed
    st.session_state.last_used_models = used

    return response_text


init_state()

NODE_ICONS = {
    "classifier": "[CLASSIFY]",
    "knn_router": "[ROUTE]",
    "hitl": "[CLARIFY]",
    "worker": "[GENERATE]",
    "parallel_workers": "[PARALLEL]",
    "aggregator": "[MERGE]",
    "judge": "[JUDGE]",
    "escalation_worker": "[ESCALATE]",
    "set_final": "[FINAL]",
}

left_col, right_col = st.columns([3, 2])

with st.sidebar:
    st.title("SwitchBoard")
    st.markdown("---")

    total_queries = max(0, len(st.session_state.messages) // 2)
    avg_latency = st.session_state.total_latency / total_queries if total_queries else 0.0

    st.metric("Total Queries", total_queries)
    st.metric("Total Cost", f"${st.session_state.total_cost:.5f}")
    st.metric(f"Total Saved vs {BASELINE_LABEL}", f"${st.session_state.total_saved:.5f}")
    st.metric("Avg Latency", f"{avg_latency:.2f}s")

    with st.expander("Available Models"):
        try:
            res = requests.get(f"{API_URL}/models", timeout=3)
            if res.status_code == 200:
                data = res.json()
                st.write("Routing Models:")
                for model_name in data.get("prototypes", []):
                    st.write(f"- `{model_name}`")
        except Exception:
            st.error("API offline.")

    st.link_button("View LangSmith", "https://smith.langchain.com")

with right_col:
    st.header("Agent Activity")

    if st.session_state.current_trace:
        for trace in st.session_state.current_trace:
            node = trace.get("node", "unknown")
            icon = NODE_ICONS.get(node, f"[{node.upper()}]")
            st.info(f"**{icon}** {trace.get('action', '')}\n\n_{trace.get('detail', '')}_")

    if st.session_state.last_routed_models or st.session_state.last_used_models:
        st.write("---")
        st.subheader("Last Model Flow")
        st.write(f"Routed: `{', '.join(st.session_state.last_routed_models) or 'N/A'}`")
        st.write(f"Used: `{', '.join(st.session_state.last_used_models) or 'N/A'}`")

    if st.session_state.knn_scores:
        st.write("---")
        st.subheader("KNN Similarity Scores")
        st.bar_chart(st.session_state.knn_scores)

with left_col:
    st.header("Chat")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if st.session_state.waiting_for_clarification:
        user_answer = st.text_input("Clarification required:", key="clarification_input")
        if st.button("Submit Clarification"):
            st.session_state.messages.append({"role": "user", "content": user_answer})
            with st.spinner("Resuming graph..."):
                try:
                    payload = {"session_id": st.session_state.session_id, "answer": user_answer}
                    response = requests.post(f"{API_URL}/resume", json=payload, stream=True)
                    st.session_state.waiting_for_clarification = False

                    final_payload, interrupt_question = parse_sse_stream(response)
                    if final_payload:
                        content = render_final_message(final_payload)
                        st.session_state.messages.append({"role": "assistant", "content": content})
                    elif interrupt_question:
                        st.session_state.messages.append({"role": "assistant", "content": interrupt_question})
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to resume API: {exc}")
                    st.session_state.waiting_for_clarification = False
    else:
        if query := st.chat_input("Ask anything..."):
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            st.session_state.current_trace = []
            st.session_state.knn_scores = {}
            st.session_state.last_routed_models = []
            st.session_state.last_used_models = []

            with st.chat_message("assistant"):
                placeholder = st.empty()
                placeholder.markdown("Gathering agents...")
                try:
                    payload = {"query": query, "session_id": st.session_state.session_id}
                    response = requests.post(f"{API_URL}/chat", json=payload, stream=True)
                    final_payload, interrupt_question = parse_sse_stream(response)

                    if final_payload:
                        content = render_final_message(final_payload)
                        placeholder.markdown(content)
                        st.session_state.messages.append({"role": "assistant", "content": content})
                    elif interrupt_question:
                        placeholder.markdown(interrupt_question)
                        st.session_state.messages.append({"role": "assistant", "content": interrupt_question})
                    else:
                        placeholder.markdown("No response received.")

                    if st.session_state.waiting_for_clarification:
                        st.rerun()
                except Exception as exc:
                    err_msg = f"**Error reaching API**: {str(exc)}"
                    placeholder.markdown(err_msg)
                    st.session_state.messages.append({"role": "assistant", "content": err_msg})


            st.rerun()
