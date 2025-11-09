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
                "--toolsets=all",
                "--dynamic-toolsets",
            ],
            "transport": "stdio",
            # "env": {
            #     "GITHUB_MCP_TOKEN": os.getenv("GITHUB_MCP_TOKEN"),
            #     "GITHUB_MCP_REPO": "springleo/new-agent",  # optional default repo
            # },
            }
        }
    )

    # Ensure API key available
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
    if not os.environ["GROQ_API_KEY"]:
        log("⚠️  GROQ_API_KEY not found in environment")

    # ─────────────────────────────────────────────
    # Step 1: Try fetching tools from both servers
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
        model = ChatGroq(model="openai/gpt-oss-120b")
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
            timeout=15,
        )
        log(f"🌦️ Weather result: {weather_response['messages'][-1].content}")
    except asyncio.TimeoutError:
        log("❌ Timeout during weather invocation.")
    except Exception as e:
        log(f"❌ Weather invocation failed: {e}")

    log("🎉 Done.")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("🛑 Interrupted by user.")
