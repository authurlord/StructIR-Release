"""structir_agent — open-domain table-QA agent framework over a StructIR table lake.

Architecture (follows common tool-agent conventions: plain-function tools returning str,
OpenAI-compatible vLLM endpoint, asyncio + semaphore runner with resume):

    TableLakeSkill (structir_skill.lake)  ←  the retrieval skill (fit per source)
        │ as_tool()
    agent_loop.ReActAgent                 ←  35B chooses source + query (verbatim-first)
        │ final evidence
    reader.answer_*                       ←  fixed 35B reader per dataset protocol
        │
    runner                                ←  conditions, trajectories, summaries

Disjoint from a separate QA system: the reader is a vanilla fixed-prompt reader; there is no
sufficiency verifier / router / budget logic.
"""
__version__ = "0.1.0"
