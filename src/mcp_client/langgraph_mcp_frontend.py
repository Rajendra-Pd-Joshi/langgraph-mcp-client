import queue
import uuid

import streamlit as st

from mcp_client.langgraph_mcp_backend import (
    chatbot,
    retrieve_all_threads,
    submit_async_task,
    # FIX: run_async was missing from imports. It is needed by
    # load_conversation to call the async aget_state() method.
    run_async,
)

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="LangGraph MCP Chatbot",
    page_icon="🤖",
    layout="centered",
)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def generate_thread_id() -> str:
    """Generate a unique conversation ID."""
    return str(uuid.uuid4())


def add_thread(thread_id: str):
    """Add a thread to the session thread list."""
    if thread_id not in st.session_state.chat_threads:
        st.session_state.chat_threads.append(thread_id)


def reset_chat():
    """Create a new conversation."""
    thread_id = generate_thread_id()
    st.session_state.thread_id = thread_id
    st.session_state.message_history = []
    add_thread(thread_id)


def load_conversation(thread_id: str) -> list:
    """Load messages belonging to a conversation."""
    try:
        # FIX: Was chatbot.get_state() (sync). Since the checkpointer
        # is AsyncSqliteSaver, the sync path raises an error at runtime.
        # Must use aget_state() via run_async().
        state = run_async(
            chatbot.aget_state(
                config={"configurable": {"thread_id": thread_id}}
            )
        )
        return state.values.get("messages", [])

    except Exception as e:
        st.error(f"Could not load conversation: {e}")
        return []


def convert_messages_to_history(messages: list) -> list:
    """
    Convert LangChain messages into the simple
    Streamlit history format.
    """
    history = []

    for message in messages:

        if isinstance(message, HumanMessage):
            history.append(
                {"role": "user", "content": str(message.content)}
            )

        elif isinstance(message, AIMessage):
            # Ignore AI messages that only contain tool calls
            # with no visible text content.
            if message.content:
                history.append(
                    {"role": "assistant", "content": str(message.content)}
                )

        elif isinstance(message, ToolMessage):
            # Tool results are intentionally not shown in history.
            continue

    return history


def switch_thread(thread_id: str):
    """Switch to an existing conversation."""
    st.session_state.thread_id = thread_id
    messages = load_conversation(thread_id)
    st.session_state.message_history = convert_messages_to_history(messages)


# ============================================================
# SESSION INITIALIZATION
# ============================================================

if "message_history" not in st.session_state:
    st.session_state.message_history = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = generate_thread_id()

if "chat_threads" not in st.session_state:
    try:
        st.session_state.chat_threads = retrieve_all_threads()
    except Exception:
        st.session_state.chat_threads = []

# Make sure the current thread is always in the list.
add_thread(st.session_state.thread_id)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🤖 LangGraph MCP")
    st.caption("AI chatbot with MCP tools")
    st.divider()

    if st.button("➕ New Chat", use_container_width=True):
        reset_chat()
        st.rerun()

    st.divider()
    st.subheader("💬 My Conversations")

    if not st.session_state.chat_threads:
        st.caption("No previous conversations.")
    else:
        for thread_id in reversed(st.session_state.chat_threads):
            is_current = thread_id == st.session_state.thread_id
            label = f"● {thread_id[:8]}" if is_current else thread_id[:8]

            if st.button(label, key=f"thread_{thread_id}", use_container_width=True):
                switch_thread(thread_id)
                st.rerun()

    st.divider()
    st.caption("Current conversation")
    st.code(st.session_state.thread_id, language=None)


# ============================================================
# MAIN HEADER
# ============================================================

st.title("🤖 LangGraph MCP Chatbot")
st.caption("Powered by LangGraph + MCP")


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.message_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# CHAT INPUT
# ============================================================

user_input = st.chat_input("Ask me anything...")


# ============================================================
# PROCESS MESSAGE
# ============================================================

if user_input:

    # --------------------------------------------------------
    # Add and display user message immediately
    # --------------------------------------------------------

    st.session_state.message_history.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # --------------------------------------------------------
    # LangGraph config
    # --------------------------------------------------------

    thread_id = st.session_state.thread_id

    CONFIG = {
        "configurable": {"thread_id": thread_id},
        "metadata": {"thread_id": thread_id},
        "run_name": "chat_turn",
    }

    # --------------------------------------------------------
    # Assistant response
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        # FIX: Removed the dead `status_box = None` that was
        # declared but never used.

        # FIX: Removed `full_response = []`. st.write_stream()
        # already returns the concatenated string as ai_message,
        # so building a parallel list was redundant.

        # FIX: Collect ALL tool names used during the turn
        # instead of overwriting a single value. The old code
        # stored to st.session_state["_active_tool"] on each
        # tool message, so only the last tool was ever shown.
        tools_used: list[str] = []


        # ====================================================
        # STREAM GENERATOR
        # ====================================================

        def stream_response():

            event_queue: queue.Queue = queue.Queue()

            async def run_stream():
                try:
                    async for message_chunk, metadata in chatbot.astream(
                        {"messages": [HumanMessage(content=user_input)]},
                        config=CONFIG,
                        stream_mode="messages",
                    ):
                        event_queue.put(("message", message_chunk, metadata))

                except Exception as exc:
                    event_queue.put(("error", exc, None))

                finally:
                    event_queue.put(("done", None, None))

            submit_async_task(run_stream())

            # Consume events from the async loop
            while True:
                event_type, payload, metadata = event_queue.get()

                if event_type == "done":
                    break

                if event_type == "error":
                    raise payload

                message_chunk = payload

                # --------------------------------------------
                # Tool message — record name, do not stream
                # --------------------------------------------

                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", None) or "tool"
                    # FIX: Append to the list so every tool call
                    # is captured, not just the last one.
                    tools_used.append(tool_name)
                    continue

                # --------------------------------------------
                # AI message — stream text content only
                # --------------------------------------------

                if isinstance(message_chunk, AIMessage):
                    content = message_chunk.content

                    if isinstance(content, str):
                        if content:
                            yield content

                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict):
                                text = block.get("text", "")
                                if text:
                                    yield text


        # ====================================================
        # STREAM AND CAPTURE RESPONSE
        # ====================================================

        try:
            ai_message = st.write_stream(stream_response())

        except Exception as e:
            st.error(f"❌ Error: {e}")
            ai_message = ""

        # ====================================================
        # TOOL STATUS — shown after streaming finishes
        # FIX: Show all tools used, not just the last one.
        # ====================================================

        if tools_used:
            tool_labels = ", ".join(f"`{name}`" for name in tools_used)
            st.caption(f"🔧 Tools used: {tool_labels}")

    # --------------------------------------------------------
    # Save assistant response to history
    # --------------------------------------------------------

    if ai_message:
        st.session_state.message_history.append(
            {"role": "assistant", "content": str(ai_message)}
        )