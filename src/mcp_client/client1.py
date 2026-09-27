import asyncio
import json

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI


load_dotenv()


SERVERS = {
    "math": {
        "transport": "stdio",
        "command": "C:\\Users\\rajen\\AppData\\Roaming\\Python\\Python313\\Scripts\\uv.exe",
        "args": [
            "--directory",
            "C:\\Users\\rajen\\Desktop\\MCP-Math-Server",
            "run",
            "fastmcp",
            "run",
            "src\\mcp_math_server\\main.py"
        ]
    },

    "expense": {
        "transport": "streamable_http",
        "url": "https://magic-beige-goat.fastmcp.app/mcp"
    }
}

async def main():

    print("Connecting to MCP servers...")

    client = MultiServerMCPClient(SERVERS)

    tools = await client.get_tools()

    print("\nAvailable MCP tools:")

    for tool in tools:
        print(f"  - {tool.name}")

    llm = ChatOpenAI(
        model="gpt-4o-mini"
    )

    llm_with_tools = llm.bind_tools(tools)

    prompt = "What is 25 multiplied by 12?"

    print("\nUser:", prompt)

    response = await llm_with_tools.ainvoke(prompt)

    print("\nLLM tool calls:")

    if response.tool_calls:
        for tool_call in response.tool_calls:
            print(
                f"  Tool: {tool_call['name']}"
            )
            print(
                f"  Arguments: {tool_call.get('args', {})}"
            )

    else:
        print("No tool call was generated.")
        print("\nLLM:", response.content)
        return

    tool_messages = []

    for tool_call in response.tool_calls:

        tool_name = tool_call["name"]
        tool_args = tool_call.get("args") or {}
        tool_call_id = tool_call["id"]

        print(f"\nExecuting: {tool_name}")

        selected_tool = next(
            tool
            for tool in tools
            if tool.name == tool_name
        )

        result = await selected_tool.ainvoke(tool_args)

        print("Tool result:", result)

        tool_messages.append(
            ToolMessage(
                tool_call_id=tool_call_id,
                content=json.dumps(result),
            )
        )

    final_response = await llm_with_tools.ainvoke(
        [
            response,
            *tool_messages,
        ]
    )

    print("\nFinal response:")
    print(final_response.content)


if __name__ == "__main__":
    asyncio.run(main())