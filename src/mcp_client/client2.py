
import asyncio
import json

import streamlit as st
from dotenv import load_dotenv

from fastmcp import Client

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from langchain_core.tools import StructuredTool


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# SERVER CONFIGURATION
# ============================================================

MATH_SERVER = {
    "mcpServers": {
        "math": {
            "transport": "stdio",
            "command": (
                r"C:\Users\rajen\AppData\Roaming\Python"
                r"\Python313\Scripts\uv.exe"
            ),
            "args": [
                "--directory",
                r"C:\Users\rajen\Desktop\MCP-Math-Server",
                "run",
                "fastmcp",
                "run",
                r"src\mcp_math_server\main.py",
            ],
        }
    }
}


EXPENSE_SERVER_URL = (
    "https://magic-beige-goat.fastmcp.app/mcp"
)


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="MCP Chat",
    page_icon="🧰",
    layout="centered",
)

st.title("🧰 MCP Chat")
st.caption("Local Math MCP + Authenticated Horizon Expense MCP")


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an MCP-powered assistant.

You have access to:

1. A local Math MCP server.
2. An authenticated Expense MCP server.

Use MCP tools whenever they are needed.

Do not narrate tool execution.

Do not say:
- "I am checking..."
- "Let me calculate..."
- "I am querying..."
- "Please wait..."

After tools finish, provide only the useful final answer.

For expense questions:
- Use the actual Expense MCP tools.
- Never invent expense data.

For mathematical questions:
- Prefer the Math MCP tools when available.
"""


# ============================================================
# CREATE MCP CLIENT
# ============================================================

def create_math_client():
    return Client(MATH_SERVER)


def create_expense_client():
    return Client(
        EXPENSE_SERVER_URL,
        auth="oauth",
    )


# ============================================================
# GET MCP TOOLS
# ============================================================

async def get_mcp_tools():
    """
    Connect to both MCP servers using the required
    async context manager and retrieve their tools.
    """

    math_client = create_math_client()
    expense_client = create_expense_client()

    # --------------------------------------------------------
    # IMPORTANT:
    # FastMCP requires async with client:
    # --------------------------------------------------------

    async with math_client:
        math_tools = await math_client.list_tools()

    async with expense_client:
        expense_tools = await expense_client.list_tools()

    return math_tools, expense_tools


# ============================================================
# MCP TOOL EXECUTOR
# ============================================================

async def execute_mcp_tool(
    server_name,
    tool_name,
    arguments,
):
    """
    Open a fresh MCP connection for each tool call.

    This avoids keeping an async context alive across
    Streamlit reruns.
    """

    if server_name == "math":

        client = create_math_client()

    elif server_name == "expense":

        client = create_expense_client()

    else:

        raise ValueError(
            f"Unknown MCP server: {server_name}"
        )


    async with client:

        result = await client.call_tool(
            tool_name,
            arguments,
        )

        return result


# ============================================================
# CONVERT MCP TOOL TO LANGCHAIN TOOL
# ============================================================

def make_langchain_tool(
    server_name,
    mcp_tool,
):
    """
    Convert an MCP tool into a LangChain StructuredTool.
    """

    async def tool_function(**kwargs):

        result = await execute_mcp_tool(
            server_name,
            mcp_tool.name,
            kwargs,
        )

        # ----------------------------------------------------
        # Extract FastMCP result content
        # ----------------------------------------------------

        if hasattr(result, "content"):

            output = []

            for item in result.content:

                if hasattr(item, "text"):

                    output.append(item.text)

                else:

                    output.append(str(item))

            return "\n".join(output)

        return str(result)


    # --------------------------------------------------------
    # MCP JSON schema -> LangChain args schema
    # --------------------------------------------------------

    return StructuredTool.from_function(
        coroutine=tool_function,
        name=mcp_tool.name,
        description=(
            mcp_tool.description
            or f"MCP tool: {mcp_tool.name}"
        ),
        args_schema=mcp_tool.inputSchema,
    )


# ============================================================
# INITIALIZE APPLICATION
# ============================================================

if "initialized" not in st.session_state:

    with st.spinner(
        "Connecting to MCP servers..."
    ):

        math_tools, expense_tools = asyncio.run(
            get_mcp_tools()
        )


    # --------------------------------------------------------
    # Convert tools
    # --------------------------------------------------------

    langchain_tools = []


    for tool in math_tools:

        langchain_tools.append(
            make_langchain_tool(
                "math",
                tool,
            )
        )


    for tool in expense_tools:

        langchain_tools.append(
            make_langchain_tool(
                "expense",
                tool,
            )
        )


    # --------------------------------------------------------
    # Store tools
    # --------------------------------------------------------

    st.session_state.tools = langchain_tools


    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    st.session_state.llm = ChatOpenAI(
        model="gpt-4o-mini",
        max_tokens=1000,
    )


    # --------------------------------------------------------
    # Bind tools
    # --------------------------------------------------------

    st.session_state.llm_with_tools = (
        st.session_state.llm.bind_tools(
            st.session_state.tools
        )
    )


    # --------------------------------------------------------
    # Conversation history
    # --------------------------------------------------------

    st.session_state.history = [
        SystemMessage(
            content=SYSTEM_PROMPT
        )
    ]


    st.session_state.initialized = True


# ============================================================
# DISPLAY PREVIOUS MESSAGES
# ============================================================

for msg in st.session_state.history:

    if isinstance(msg, HumanMessage):

        with st.chat_message("user"):
            st.markdown(msg.content)


    elif isinstance(msg, AIMessage):

        # Don't display intermediate tool calls
        if getattr(msg, "tool_calls", None):
            continue

        if msg.content:

            with st.chat_message("assistant"):
                st.markdown(msg.content)


# ============================================================
# CHAT INPUT
# ============================================================

user_text = st.chat_input(
    "Ask something..."
)


if user_text:

    # --------------------------------------------------------
    # Display user message
    # --------------------------------------------------------

    with st.chat_message("user"):
        st.markdown(user_text)


    # --------------------------------------------------------
    # Store user message
    # --------------------------------------------------------

    st.session_state.history.append(
        HumanMessage(
            content=user_text
        )
    )


    # ========================================================
    # FIRST LLM CALL
    # ========================================================

    first_response = asyncio.run(
        st.session_state.llm_with_tools.ainvoke(
            st.session_state.history
        )
    )


    # ========================================================
    # CHECK TOOL CALLS
    # ========================================================

    tool_calls = getattr(
        first_response,
        "tool_calls",
        None,
    )


    # ========================================================
    # NORMAL RESPONSE
    # ========================================================

    if not tool_calls:

        with st.chat_message("assistant"):

            st.markdown(
                first_response.content or ""
            )


        st.session_state.history.append(
            first_response
        )


    # ========================================================
    # MCP TOOL CALL
    # ========================================================

    else:

        # ----------------------------------------------------
        # Store AI tool-call message
        # ----------------------------------------------------

        st.session_state.history.append(
            first_response
        )


        # ----------------------------------------------------
        # Execute tools
        # ----------------------------------------------------

        for tool_call in tool_calls:

            tool_name = tool_call["name"]

            tool_args = tool_call.get(
                "args",
                {},
            )


            # ------------------------------------------------
            # Find LangChain tool
            # ------------------------------------------------

            selected_tool = None

            for tool in st.session_state.tools:

                if tool.name == tool_name:

                    selected_tool = tool
                    break


            if selected_tool is None:

                result_text = json.dumps(
                    {
                        "error": (
                            f"Tool '{tool_name}' "
                            "was not found."
                        )
                    }
                )

            else:

                try:

                    tool_result = asyncio.run(
                        selected_tool.ainvoke(
                            tool_args
                        )
                    )


                    # ----------------------------------------
                    # Normalize result
                    # ----------------------------------------

                    if isinstance(
                        tool_result,
                        str,
                    ):

                        result_text = tool_result

                    else:

                        result_text = json.dumps(
                            tool_result,
                            default=str,
                        )


                except Exception as e:

                    result_text = json.dumps(
                        {
                            "error": str(e)
                        }
                    )


            # ------------------------------------------------
            # Add MCP result
            # ------------------------------------------------

            st.session_state.history.append(
                ToolMessage(
                    content=result_text,
                    tool_call_id=tool_call["id"],
                )
            )


        # ====================================================
        # FINAL LLM RESPONSE
        # ====================================================

        final_response = asyncio.run(
            st.session_state.llm.ainvoke(
                st.session_state.history
            )
        )


        # ----------------------------------------------------
        # Display final answer
        # ----------------------------------------------------

        with st.chat_message("assistant"):

            st.markdown(
                final_response.content or ""
            )


        # ----------------------------------------------------
        # Store final answer
        # ----------------------------------------------------

        st.session_state.history.append(
            AIMessage(
                content=(
                    final_response.content
                    or ""
                )
            )
        )

