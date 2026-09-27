
import asyncio
from fastmcp import Client


SERVER_URL = "https://magic-beige-goat.fastmcp.app/mcp"


async def main():
    client = Client(
        SERVER_URL,
        auth="oauth",
    )

    async with client:
        print("Connected to Horizon Expense MCP!")

        await client.ping()

        tools = await client.list_tools()

        print("\nAvailable tools:")
        for tool in tools:
            print(f"  - {tool.name}")

        resources = await client.list_resources()

        print("\nAvailable resources:")
        for resource in resources:
            print(f"  - {resource.uri}")


if __name__ == "__main__":
    asyncio.run(main())

