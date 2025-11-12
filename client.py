#!/usr/bin/env python3
import asyncio
import os
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq

load_dotenv()

# ─────────────────────────────────────────────
# Utility: simple async-safe print
def log(msg):
    print(f"[client] {msg}", flush=True)


# ─────────────────────────────────────────────
async def main():
    log("Initializing MultiServerMCPClient...")

    client = MultiServerMCPClient(
        {
            # math server via stdio (note: -u for unbuffered mode!)
            "math": {
                "command": "python",
                "args": ["-u", "mathserver.py"],
                "transport": "stdio",
            },
            # weather server via HTTP
            "weather": {
                "url": "http://127.0.0.1:8000/mcp",
                "transport": "streamable_http",
            },
            "github": {
                "command": "./github-mcp-server/github-mcp-server",
                "args": [
                    "stdio",
                    "--toolsets=issues,pull_requests,users,orgs,actions",
                    "--dynamic-toolsets",
                ],
                "transport": "stdio",
                "env": {
                    "GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN"),
                    "GITHUB_MCP_REPO": "springleo/new-agent",  # optional default repo
                },
            }
        }
    )

    # Ensure API key available
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
    if not os.environ["GROQ_API_KEY"]:
        log("⚠️  GROQ_API_KEY not found in environment")

    # ─────────────────────────────────────────────
    # Step 1: Try fetching tools from all servers
    log("Fetching available MCP tools...")
    try:
        tools = await asyncio.wait_for(client.get_tools(), timeout=10)
        log(f"✅ MCP tools loaded: {[t.name for t in tools]}")
    except asyncio.TimeoutError:
        log("❌ Timeout: mathserver may not be responding via stdio")
        return
    except Exception as e:
        log(f"❌ Failed to get tools: {e}")
        return

    # ─────────────────────────────────────────────
    # Step 2: Setup model & agent
    try:
        model = ChatGroq(model="gpt-4o-mini")
        agent = create_react_agent(model, tools)
        log("✅ LangGraph agent initialized with MCP tools")
    except Exception as e:
        log(f"❌ Failed to create agent: {e}")
        return

    # ─────────────────────────────────────────────
    # Step 3: Invoke math
    try:
        log("Invoking math tool...")
        math_response = await asyncio.wait_for(
            agent.ainvoke(
                {
                    "messages": [
                        {"role": "user", "content": "What is 4 times 5 plus 20?"},
                    ]
                }
            ),
            timeout=15,
        )
        log(f"📐 Math result: {math_response['messages'][-1].content}")
    except asyncio.TimeoutError:
        log("❌ Timeout during math invocation.")
    except Exception as e:
        log(f"❌ Math invocation failed: {e}")

    # ─────────────────────────────────────────────
    # Step 4: Invoke weather
    try:
        log("Invoking weather tool...")
        weather_response = await asyncio.wait_for(
            agent.ainvoke(
                {
                    "messages": [
                        {"role": "user", "content": "What is weather in Pune?"},
                    ]
                }
            ),
            timeout=20,
        )
        log(f"🌦️ Weather result: {weather_response['messages'][-1].content}")
    except asyncio.TimeoutError:
        log("❌ Timeout during weather invocation.")
    except Exception as e:
        log(f"❌ Weather invocation failed: {e}")

  # ─────────────────────────────────────────────
    # Step 4: Invoke github
    try:
        log("Invoking github tool...")
        github_response = await asyncio.wait_for(
            agent.ainvoke(
                {
                    "messages": [
                        {"role": "user", "content": "are there any open issues or prs in this repo 'springleo/new-agent' ? Also check if there is a CI workflow present in this repo"},
                    ],
                }
            ),
            timeout=15,
        )
        log(f"🌦️ github result: {github_response['messages'][-1].content}")
    except asyncio.TimeoutError:
        log("❌ Timeout during github invocation.")
    except Exception as e:
        log(f"❌ github invocation failed: {e}")
    # # FOR Github MCP srv
    # # Step 1: Access the underlying GitHub MCP client
    # client.client       
    # github_client = client.clients["github"]

    # # Step 2: List all available toolsets
    # print("\n[client] Listing available GitHub toolsets...")
    # toolsets = await github_client.call_tool(
    #     "list_available_toolsets",  # tool name
    #     {}                           # tool input (empty dict)
    # )
    # print("Available toolsets:", toolsets)

    # # Step 3: Enable a toolset
    # print("\n[client] Enabling 'actions' toolset...")
    # await github_client.call_tool("enable_toolset", {"toolset_name": "actions"})

    # # Step 4: Get tools within that toolset
    # print("\n[client] Fetching tools under 'actions' toolset...")
    # actions_tools = await github_client.call_tool("get_toolset_tools", {"toolset_name": "actions"})
    # print("Actions tools:", actions_tools)

    # print("\n[client] Asking agent to trigger workflow...")
    # response = await agent.ainvoke({
    #     "messages": [
    #         {
    #             "role": "user",
    #             "content": "Is there a CI workflow in this repo ?",
    #         }
    #     ]
    # })
    # print("\n[client] Response:\n", response["messages"][-1].content)
    log("🎉 Done.")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("🛑 Interrupted by user.")
