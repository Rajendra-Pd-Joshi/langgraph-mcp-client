from dotenv import load_dotenv
load_dotenv()
import asyncio
import json
from pathlib import Path

import streamlit as st

from fastmcp import Client
from fastmcp.client.auth import OAuth

from langchain_mcp_adapters.client import MultiServerMCPClient

from langchain_openai import ChatOpenAI

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)

from langchain_core.tools import StructuredTool
from pydantic import create_model


# ============================================================
# CONFIGURATION
# ============================================================

EXPENSE_MCP_URL = (
    "https://magic-beige-goat.fastmcp.app/mcp"
)

CALLBACK_PORT = 53030


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

HORIZON_CLIENT_FILE = (
    BASE_DIR / "horizon_client.json"
)

MATH_SERVER_DIR = (
    r"C:\Users\rajen\Desktop\MCP-Math-Server"
)

UV_PATH = (
    r"C:\Users\rajen\AppData\Roaming"
    r"\Python\Python313\Scripts\uv.exe"
)


# ============================================================
# LOAD HORIZON CREDENTIALS
# ============================================================

def load_horizon_credentials():

    if not HORIZON_CLIENT_FILE.exists():

        raise FileNotFoundError(
            f"\nCould not find:\n"
            f"{HORIZON_CLIENT_FILE}\n\n"
            "Make sure horizon_client.json exists."
        )

    with open(
        HORIZON_CLIENT_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        credentials = json.load(f)

    if not credentials.get("client_id"):

        raise ValueError(
            "client_id is missing from "
            "horizon_client.json"
        )

    if not credentials.get("client_secret"):

        raise ValueError(
            "client_secret is missing from "
            "horizon_client.json"
        )

    return credentials


# ============================================================
# CREATE REMOTE EXPENSE MCP CLIENT
# ============================================================

def create_expense_client():

    credentials = load_horizon_credentials()

    oauth = OAuth(
        mcp_url=EXPENSE_MCP_URL,

        client_name=(
            "Rajendra Expense MCP Client"
        ),

        client_id=credentials["client_id"],

        client_secret=credentials["client_secret"],

        callback_port=CALLBACK_PORT,
    )

    return Client(
        EXPENSE_MCP_URL,
        auth=oauth,
    )


# ============================================================
# CREATE LOCAL MATH MCP CLIENT
# ============================================================

def create_math_client():

    servers = {

        "math": {

            "transport": "stdio",

            "command": UV_PATH,

            "args": [

                "--directory",

                MATH_SERVER_DIR,

                "run",

                "fastmcp",

                "run",

                r"src\mcp_math_server\main.py",
            ],
        }
    }

    return MultiServerMCPClient(
        servers
    )


# ============================================================
# CONVERT REMOTE MCP TOOL TO LANGCHAIN TOOL
# ============================================================

def create_langchain_expense_tool(
    expense_client,
    tool_info,
):

    # --------------------------------------------------------
    # Get MCP input schema
    # --------------------------------------------------------

    schema = getattr(
        tool_info,
        "inputSchema",
        None,
    )

    if schema is None:

        schema = getattr(
            tool_info,
            "input_schema",
            None,
        )

    if schema is None:

        schema = {}

    properties = schema.get(
        "properties",
        {},
    )

    required_fields = schema.get(
        "required",
        [],
    )

    fields = {}

    # --------------------------------------------------------
    # Convert JSON schema -> Pydantic fields
    # --------------------------------------------------------

    for field_name, field_info in properties.items():

        field_type = field_info.get(
            "type",
            "string",
        )

        if field_type == "integer":

            python_type = int

        elif field_type == "number":

            python_type = float

        elif field_type == "boolean":

            python_type = bool

        elif field_type == "array":

            python_type = list

        elif field_type == "object":

            python_type = dict

        else:

            python_type = str

        # Required field

        if field_name in required_fields:

            fields[field_name] = (
                python_type,
                ...,
            )

        # Optional field

        else:

            fields[field_name] = (
                python_type | None,
                None,
            )

    # --------------------------------------------------------
    # Create Pydantic input model
    # --------------------------------------------------------

    ToolInput = create_model(
        f"{tool_info.name}Input",
        **fields,
    )

    # --------------------------------------------------------
    # Async tool function
    # --------------------------------------------------------

    async def call_expense_tool(
        **kwargs,
    ):

        result = await expense_client.call_tool(
            tool_info.name,
            kwargs,
        )

        # ----------------------------------------------------
        # MCP CallToolResult
        # ----------------------------------------------------

        if hasattr(
            result,
            "content",
        ):

            output = []

            for item in result.content:

                if hasattr(
                    item,
                    "text",
                ):

                    output.append(
                        item.text
                    )

                else:

                    output.append(
                        str(item)
                    )

            return "\n".join(
                output
            )

        return str(result)

    # --------------------------------------------------------
    # LangChain StructuredTool
    # --------------------------------------------------------

    return StructuredTool.from_function(
        coroutine=call_expense_tool,

        name=tool_info.name,

        description=(
            getattr(
                tool_info,
                "description",
                None,
            )
            or
            f"Expense MCP tool: {tool_info.name}"
        ),

        args_schema=ToolInput,
    )


# ============================================================
# INITIALIZE ALL MCP TOOLS
# ============================================================

async def initialize_mcp():

    # ========================================================
    # LOCAL MATH MCP
    # ========================================================

    math_client = create_math_client()

    math_tools = await math_client.get_tools()

    print(
        f"Loaded {len(math_tools)} Math MCP tools"
    )

    # ========================================================
    # REMOTE EXPENSE MCP
    # ========================================================

    expense_client = create_expense_client()

    # --------------------------------------------------------
    # IMPORTANT
    # Enter FastMCP client context
    # --------------------------------------------------------

    await expense_client.__aenter__()

    try:

        await expense_client.ping()

        print(
            "Connected to Remote Expense MCP"
        )

        # ----------------------------------------------------
        # Get remote tools
        # ----------------------------------------------------

        expense_tools_info = (
            await expense_client.list_tools()
        )

        print(
            f"Loaded "
            f"{len(expense_tools_info)} "
            f"Expense MCP tools"
        )

        # ----------------------------------------------------
        # Convert tools
        # ----------------------------------------------------

        expense_tools = []

        for tool_info in expense_tools_info:

            tool = create_langchain_expense_tool(
                expense_client,
                tool_info,
            )

            expense_tools.append(
                tool
            )

        # ----------------------------------------------------
        # Return everything
        # ----------------------------------------------------

        return {
            "math_tools": math_tools,

            "expense_tools": expense_tools,

            "all_tools": (
                math_tools
                + expense_tools
            ),

            "math_client": math_client,

            "expense_client": expense_client,
        }

    except Exception:

        await expense_client.__aexit__(
            None,
            None,
            None,
        )

        raise


# ============================================================
# STREAMLIT CACHED MCP CONNECTION
# ============================================================

@st.cache_resource(
    show_spinner="Connecting to MCP servers..."
)
def initialize_mcp_sync():

    return asyncio.run(
        initialize_mcp()
    )


# ============================================================
# LOAD MCP
# ============================================================

try:

    mcp_data = initialize_mcp_sync()

    MCP_TOOLS = mcp_data["all_tools"]

except Exception as e:

    st.error(
        "Failed to connect to MCP servers."
    )

    st.exception(e)

    st.stop()


# ============================================================
# LLM
# ============================================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)


# ============================================================
# LLM WITH MCP TOOLS
# ============================================================

llm_with_tools = llm.bind_tools(
    MCP_TOOLS
)


# ============================================================
# STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="MCP AI Assistant",
    page_icon="🤖",
    layout="centered",
)


st.title(
    "🤖 MCP AI Assistant"
)

st.caption(
    "GPT-4o-mini + Local Math MCP + Remote Expense MCP"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "MCP Servers"
    )

    st.success(
        "🧮 Math MCP"
    )

    st.success(
        "💰 Expense MCP"
    )

    st.divider()

    st.write(
        f"Connected tools: {len(MCP_TOOLS)}"
    )

    st.write(
        "Remote server:"
    )

    st.code(
        EXPENSE_MCP_URL
    )


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = [

        SystemMessage(
            content=(
                "You are a helpful AI assistant "
                "with access to two MCP servers.\n\n"

                "Math MCP provides mathematical "
                "operations such as addition, "
                "subtraction, multiplication, "
                "division and modulo.\n\n"

                "Expense MCP provides expense "
                "tracking functionality.\n\n"

                "Use MCP tools whenever they are "
                "appropriate for the user's request."
            )
        )
    ]


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    if isinstance(
        message,
        HumanMessage,
    ):

        with st.chat_message(
            "user"
        ):

            st.write(
                message.content
            )

    elif isinstance(
        message,
        AIMessage,
    ):

        if message.content:

            with st.chat_message(
                "assistant"
            ):

                st.write(
                    message.content
                )


# ============================================================
# CHAT INPUT
# ============================================================

user_input = st.chat_input(
    "Ask me anything..."
)


# ============================================================
# PROCESS USER MESSAGE
# ============================================================

if user_input:

    # --------------------------------------------------------
    # Display user message
    # --------------------------------------------------------

    with st.chat_message(
        "user"
    ):

        st.write(
            user_input
        )

    # --------------------------------------------------------
    # Add to history
    # --------------------------------------------------------

    st.session_state.messages.append(
        HumanMessage(
            content=user_input
        )
    )


    # ========================================================
    # ASYNC AGENT
    # ========================================================

    async def process_message():

        # ----------------------------------------------------
        # First LLM call
        # ----------------------------------------------------

        response = (
            await llm_with_tools.ainvoke(
                st.session_state.messages
            )
        )

        # ----------------------------------------------------
        # No tools needed
        # ----------------------------------------------------

        if not response.tool_calls:

            return response


        # ----------------------------------------------------
        # Save AI tool-call message
        # ----------------------------------------------------

        st.session_state.messages.append(
            response
        )


        # ----------------------------------------------------
        # Execute every requested tool
        # ----------------------------------------------------

        for tool_call in response.tool_calls:

            tool_name = tool_call["name"]

            tool_args = tool_call.get(
                "args",
                {},
            )

            # Find LangChain MCP tool

            selected_tool = None

            for tool in MCP_TOOLS:

                if tool.name == tool_name:

                    selected_tool = tool

                    break

            # ------------------------------------------------
            # Tool not found
            # ------------------------------------------------

            if selected_tool is None:

                tool_result = (
                    f"Tool '{tool_name}' "
                    "was not found."
                )

            else:

                try:

                    tool_result = (
                        await selected_tool.ainvoke(
                            tool_args
                        )
                    )

                except Exception as e:

                    tool_result = (
                        f"Tool execution failed: {e}"
                    )

            # ------------------------------------------------
            # Add tool result
            # ------------------------------------------------

            st.session_state.messages.append(

                ToolMessage(

                    content=str(
                        tool_result
                    ),

                    tool_call_id=(
                        tool_call["id"]
                    ),
                )
            )


        # ----------------------------------------------------
        # Final LLM response
        # ----------------------------------------------------

        final_response = (
            await llm_with_tools.ainvoke(
                st.session_state.messages
            )
        )

        return final_response


    # ========================================================
    # EXECUTE
    # ========================================================

    try:

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "Thinking..."
            ):

                final_response = (
                    asyncio.run(
                        process_message()
                    )
                )

            st.write(
                final_response.content
            )

        # ----------------------------------------------------
        # Save final response
        # ----------------------------------------------------

        st.session_state.messages.append(
            final_response
        )

    except Exception as e:

        st.error(
            "Error while processing request:"
        )

        st.exception(e)

