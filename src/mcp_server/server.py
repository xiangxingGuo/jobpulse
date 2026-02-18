from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from .tools_fetch import fetch_jd
from .tools_extract import extract_local
from .tools_qc import qc_validate
from .tools_report import generate_report_api

mcp = FastMCP("jobpulse")

# Register tools (decorators can be defined in respective files; explicit registration here for clarity)
mcp.tool()(fetch_jd)
mcp.tool()(extract_local)
mcp.tool()(qc_validate)
mcp.tool()(generate_report_api)

def main():
    # stdio: the most common local process method (Cursor/Claude Desktop/Agent SDK are also commonly used)
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
