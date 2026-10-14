# MCP Server calls the query pipeline in-process and returns full Articles

Status: accepted

The query pipeline could only be reached through the REST API, so an MCP host such as Claude Code
couldn't use it as a tool. We added an MCP Server, and two decisions shaped it.

**1. The MCP Server calls the pipeline in-process, and its HTTP transport lives in the API's app.**
The `/query` logic (Query Language gate, Hybrid Retrieval, fusion, reranking, generation) was
extracted from the FastAPI route handler into one framework-free function, `answer_query`. The REST
route and the MCP tools are both thin adapters over it, so the two interfaces can't drift apart.
The MCP Server is built by a factory that receives *providers* of its components, resolved on every
tool call: building the server loads nothing, the first tool call pays for model loading, and the
cached providers keep the models warm afterwards. Over stdio, Claude Code runs the server as a
subprocess. Over Streamable HTTP, the server is mounted inside the existing FastAPI app at `/mcp`,
built with the same cached providers as the REST routes, so one process and one container serve
both and load the models once.

**2. `ask_code_civil` returns the generated answer plus the full text of the Retrieved Articles.**
Unlike `/query`, whose per-Article entries are only `ref` + `sectionParentTitre`, each Retrieved
Article comes back with its complete `texte`. The host is itself a model that can read the sources,
so it can check the answer against them instead of trusting it blindly. This follows ADR 0001: what
the host sees is exactly what the generation prompt saw, full Articles and never Chunks. A second
tool, `get_article(ref)`, returns any Article by `ref`, so the host can follow a reference the
answer makes to an Article that wasn't retrieved (e.g. "voir article 1240").

## Considered options

- **Wrap the REST API (the MCP Server calls `/query` over HTTP).** Rejected: it adds a hop and a
  second process to run, the stdio server would need the API to be up first, and `/query` doesn't
  return Article text, so the server would need one extra `/articles/{ref}` call per Retrieved
  Article.
- **Run the HTTP MCP Server as its own process or compose service.** Rejected: it would load a
  second copy of the embedding model and Reranker, doubling memory, for no isolation benefit on a
  localhost-only tool.
- **Return the answer only, or the answer and `ref`s, from `ask_code_civil`.** Rejected: the host
  couldn't verify the answer, or would need one `get_article` call per Retrieved Article to do so.
  The extra tokens of full text are the point of the tool.
