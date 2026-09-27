from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from typing import (
    TypedDict,
    Annotated,
    Optional,
)

from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
)

from langchain_openai import ChatOpenAI

from langgraph.checkpoint.sqlite.aio import (
    AsyncSqliteSaver,
)

from langgraph.graph.message import (
    add_messages,
)

from langgraph.prebuilt import (
    ToolNode,
    tools_condition,
)

from langchain_community.tools import (
    DuckDuckGoSearchRun,
)

from langchain_core.tools import (
    tool,
    StructuredTool,
)

from langchain_mcp_adapters.client import (
    MultiServerMCPClient,
)

from fastmcp import Client
from fastmcp.client.auth import OAuth

# FIX: Moved pydantic import to top level — it was buried inside
# create_expense_langchain_tool(), which re-imports on every call.
from pydantic import create_model

from dotenv import load_dotenv

import aiosqlite
import requests
import asyncio
import threading
import json
from pathlib import Path


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# DEDICATED ASYNC EVENT LOOP
# ============================================================

_ASYNC_LOOP = asyncio.new_event_loop()

_ASYNC_THREAD = threading.Thread(
    target=_ASYNC_LOOP.run_forever,
    daemon=True,
)

_ASYNC_THREAD.start()


def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(
        coro,
        _ASYNC_LOOP,
    )


def run_async(coro):
    return _submit_async(coro).result()


def submit_async_task(coro):
    """Schedule a coroutine on the backend event loop."""
    return _submit_async(coro)


# ============================================================
# 1. LLM
# ============================================================

# gpt-4o is required for reliable tool selection.
# The default gpt-3.5-turbo frequently picks the wrong tool
# (e.g. web search) when a more specific MCP tool should be used.
llm = ChatOpenAI(model="gpt-4o-mini",max_tokens = 10000)


# ============================================================
# 2. NORMAL TOOLS
# ============================================================

search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch the latest stock price for a given
    stock symbol such as AAPL or TSLA.
    """
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE"
        f"&symbol={symbol}"
        f"&apikey=C9PE94QUEW9VWGFM"
    )
    response = requests.get(url, timeout=30)
    return response.json()


# ============================================================
# 3. LOCAL MATH MCP
# ============================================================

MATH_SERVER_DIR = r"C:\Users\rajen\Desktop\MCP-Math-Server"

UV_PATH = (
    r"C:\Users\rajen\AppData\Roaming"
    r"\Python\Python313\Scripts\uv.exe"
)

# FIX: Client is instantiated once. __aenter__ is called in
# _load_mcp_tools to start the stdio subprocess and keep it
# alive for the program's lifetime. Never call __aexit__ on
# this — doing so would kill the subprocess.
math_client = MultiServerMCPClient(
    {
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
)


# ============================================================
# 4. REMOTE EXPENSE MCP
# ============================================================

EXPENSE_MCP_URL = "https://magic-beige-goat.fastmcp.app/mcp"

CALLBACK_PORT = 53030

BASE_DIR = Path(__file__).resolve().parent
HORIZON_CLIENT_FILE = BASE_DIR / "horizon_client.json"


def load_horizon_credentials() -> dict:
    if not HORIZON_CLIENT_FILE.exists():
        raise FileNotFoundError(
            "\nMissing horizon_client.json\n"
            f"Expected location:\n{HORIZON_CLIENT_FILE}\n"
        )

    with open(HORIZON_CLIENT_FILE, "r", encoding="utf-8") as file:
        credentials = json.load(file)

    if not credentials.get("client_id"):
        raise ValueError("client_id missing from horizon_client.json")

    if not credentials.get("client_secret"):
        raise ValueError("client_secret missing from horizon_client.json")

    return credentials


def create_expense_client() -> Client:
    credentials = load_horizon_credentials()

    oauth = OAuth(
        mcp_url=EXPENSE_MCP_URL,
        client_name="LangGraph Expense Client",
        client_id=credentials["client_id"],
        client_secret=credentials["client_secret"],
        callback_port=CALLBACK_PORT,
    )

    return Client(EXPENSE_MCP_URL, auth=oauth)


# ============================================================
# CONVERT REMOTE MCP TOOL TO LANGCHAIN TOOL
# ============================================================

def create_expense_langchain_tool(
    expense_client: Client,
    tool_info,
) -> StructuredTool:

    schema = getattr(tool_info, "inputSchema", None)
    if schema is None:
        schema = getattr(tool_info, "input_schema", None)
    if schema is None:
        schema = {}

    properties = schema.get("properties", {})
    required_fields = schema.get("required", [])

    # Build Pydantic model dynamically
    fields: dict = {}

    for name, definition in properties.items():
        json_type = definition.get("type", "string")

        if json_type == "integer":
            python_type = int
        elif json_type == "number":
            python_type = float
        elif json_type == "boolean":
            python_type = bool
        elif json_type == "array":
            python_type = list
        elif json_type == "object":
            python_type = dict
        else:
            python_type = str

        if name in required_fields:
            fields[name] = (python_type, ...)
        else:
            fields[name] = (Optional[python_type], None)

    ToolInput = create_model(f"{tool_info.name}Input", **fields)

    async def call_tool(**kwargs):
        result = await expense_client.call_tool(tool_info.name, kwargs)

        if hasattr(result, "content"):
            output = []
            for item in result.content:
                output.append(item.text if hasattr(item, "text") else str(item))
            return "\n".join(output)

        return str(result)

    return StructuredTool.from_function(
        coroutine=call_tool,
        name=tool_info.name,
        description=(
            getattr(tool_info, "description", None)
            or f"Expense MCP tool: {tool_info.name}"
        ),
        args_schema=ToolInput,
    )


# ============================================================
# 5. LOAD ALL MCP TOOLS
# ============================================================

async def _load_mcp_tools():

    # --------------------------------------------------------
    # LOCAL MATH TOOLS
    # FIX: __aenter__ starts the stdio subprocess and must be
    # called before get_tools(). Never call __aexit__ — that
    # would terminate the subprocess mid-session.
    # --------------------------------------------------------
    await math_client.__aenter__()
    math_tools = await math_client.get_tools()

    print("\nMath MCP tools:")
    for t in math_tools:
        print(f"  - {t.name}")

    # --------------------------------------------------------
    # REMOTE EXPENSE MCP
    # FIX: Same lifecycle pattern — enter once, stay entered.
    # --------------------------------------------------------
    expense_client = create_expense_client()
    await expense_client.__aenter__()

    try:
        await expense_client.ping()
        print("\nExpense MCP connected.")

        expense_tool_info = await expense_client.list_tools()

        print("\nExpense MCP tools:")
        for t in expense_tool_info:
            print(f"  - {t.name}")

        expense_tools = [
            create_expense_langchain_tool(expense_client, t)
            for t in expense_tool_info
        ]

        return math_tools, expense_tools, expense_client

    except Exception:
        # Only exit the expense context on failure; math client
        # stays alive because it was entered before the try block.
        await expense_client.__aexit__(None, None, None)
        raise


# ============================================================
# LOAD MCP TOOLS
# ============================================================

try:
    math_tools, expense_tools, expense_client = run_async(
        _load_mcp_tools()
    )

except Exception as e:
    print("\nMCP connection failed:")
    print(e)
    math_tools = []
    expense_tools = []
    expense_client = None


# ============================================================
# 6. COMBINE ALL TOOLS
# ============================================================

tools = [
    search_tool,
    get_stock_price,
    *math_tools,
    *expense_tools,
]

print(f"\nTotal tools loaded: {len(tools)}")
for t in tools:
    print(f"  - {t.name}")


# ============================================================
# 7. LLM WITH TOOLS
# ============================================================

llm_with_tools = llm.bind_tools(tools) if tools else llm


# ============================================================
# 8. STATE
# ============================================================

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ============================================================
# 9. SYSTEM PROMPT
#
# A system prompt is critical for correct tool routing.
# Without it the LLM has no context about what tools exist or
# when to prefer them, so it falls back to web search for
# queries that should go to the expense or math MCP tools.
#
# Rules encoded here:
#   • Use expense MCP tools for ANY expense/finance request.
#   • Use math MCP tools for calculations and equations.
#   • Use web search only for general knowledge queries.
#   • Use get_stock_price only for stock price lookups.
#   • Never answer an expense or math question from memory
#     when the appropriate tool is available.
# ============================================================

SYSTEM_PROMPT = SystemMessage(content="""
You are a helpful AI assistant with access to the following tools:

1. **Expense MCP tools** — Add, update, retrieve, and manage
   personal expense records stored in the Horizon expense
   tracker. Use these for ANY request about expenses, spending,
   budgets, or financial records (e.g. "add Rs 1000 for food",
   "show all my expenses", "how much did I spend this month").
   Always prefer these tools over web search for such queries.

2. **Math MCP tools** — Perform arithmetic, algebra, or any
   numerical calculation. Use these instead of computing in
   your head when a calculation tool is available.

3. **get_stock_price** — Fetch the current price of a stock
   by its ticker symbol (e.g. AAPL, TSLA).

4. **duckduckgo_search** — Search the web for general
   knowledge, news, or topics not covered by the tools above.
   Do NOT use this for expense or math queries.

Always select the most specific tool available for the task.
""")


# ============================================================
# 10. CHAT NODE
# ============================================================

async def chat_node(state: ChatState):
    """
    LLM node that prepends the system prompt on every call
    so tool-routing rules are always in context, then either
    answers directly or requests tool execution.
    """
    messages = [SYSTEM_PROMPT, *state["messages"]]
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}


# ============================================================
# 11. TOOL NODE
# ============================================================

tool_node = ToolNode(tools) if tools else None


# ============================================================
# 12. SQLITE CHECKPOINTER
# FIX: Added await saver.setup() — required in LangGraph 0.2+
# to create the checkpoint tables before first use. Without
# this the graph raises an OperationalError on first run.
# ============================================================

async def _init_checkpointer() -> AsyncSqliteSaver:
    conn = await aiosqlite.connect(database="chatbot.db")
    saver = AsyncSqliteSaver(conn)
    await saver.setup()
    return saver


checkpointer = run_async(_init_checkpointer())


# ============================================================
# 13. LANGGRAPH
# ============================================================

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")

if tool_node:
    graph.add_node("tools", tool_node)
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")
else:
    graph.add_edge("chat_node", END)


# ============================================================
# COMPILE GRAPH
# ============================================================

chatbot = graph.compile(checkpointer=checkpointer)


# ============================================================
# 14. RETRIEVE ALL THREADS
# ============================================================

async def _alist_threads() -> list[str]:
    all_threads: set[str] = set()

    async for checkpoint in checkpointer.alist(None):
        try:
            thread_id = (
                checkpoint.config["configurable"]["thread_id"]
            )
            all_threads.add(thread_id)
        except (KeyError, TypeError):
            continue

    return list(all_threads)


def retrieve_all_threads() -> list[str]:
    return run_async(_alist_threads())