"""Run the MCP server on the stdio transport: `python -m src.code_civil_mcp`.

stdout carries the MCP protocol only. Nothing here prints, and MCPServer
configures logging to stderr when it is built.
"""

from __future__ import annotations

from src.api.dependencies import build_query_pipeline, get_article_store
from src.code_civil_mcp.server import create_server


def main() -> None:
    """Start the MCP server on stdio, with the production (cached) providers.

    Unauthenticated by design: stdio has no tokens, so no role is required.
    """
    create_server(build_query_pipeline, get_article_store, require_query_role=False).run(
        "stdio"
    )


if __name__ == "__main__":
    main()
