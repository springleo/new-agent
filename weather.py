#!/usr/bin/env python3
from mcp.server.fastmcp import FastMCP

mcp=FastMCP("Weather")

@mcp.tool()
async def get_weather(location:str)->str:
    """Get the weather location"""
    return "It's always rains in Pune"

if __name__=="__main__":
    mcp.run(transport="streamable-http")