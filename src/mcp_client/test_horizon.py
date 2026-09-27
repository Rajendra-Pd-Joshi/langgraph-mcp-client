
import asyncio
import json

from fastmcp import Client
from fastmcp.client.auth import OAuth


MCP_URL = "https://magic-beige-goat.fastmcp.app/mcp"
CALLBACK_PORT = 53030


def load_credentials():
    with open("horizon_client.json", "r", encoding="utf-8") as f:
        return json.load(f)


async def main():

    credentials = load_credentials()

    client_id = credentials["client_id"]
    client_secret = credentials["client_secret"]

    print("Using OAuth client:")
    print(client_id)
    print("Auth method: client_secret_post")

    oauth = OAuth(
        mcp_url=MCP_URL,
        client_id=client_id,
        client_secret=client_secret,
        callback_port=CALLBACK_PORT,
    )

    client = Client(
        MCP_URL,
        auth=oauth,
    )

    async with client:

        print("\nConnected to Horizon Expense MCP!")

        await client.ping()

        print("Ping successful!")

        tools = await client.list_tools()

        print("\nAvailable tools:")

        for tool in tools:
            print(f" - {tool.name}")

        resources = await client.list_resources()

        print("\nAvailable resources:")

        for resource in resources:
            print(f" - {resource.uri}")


if __name__ == "__main__":
    asyncio.run(main())

