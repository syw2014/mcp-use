"""
End-to-end integration test for agent reasoning functionality.

Tests the agent.run() and agent.stream() methods with reasoning=True parameter.
"""

import sys
from pathlib import Path

import pytest
from langchain_openai import ChatOpenAI

from mcp_use import MCPAgent, MCPClient
from mcp_use.logging import logger


@pytest.mark.asyncio
@pytest.mark.integration
async def test_agent_run_with_reasoning():
    """Test agent.run() with reasoning=True generates and outputs a plan."""
    server_path = Path(__file__).parent.parent / "servers_for_testing" / "simple_server.py"

    config = {"mcpServers": {"simple": {"command": sys.executable, "args": [str(server_path), "--transport", "stdio"]}}}

    client = MCPClient.from_dict(config)
    llm = ChatOpenAI(model="gpt-4o")
    agent = MCPAgent(llm=llm, client=client, max_steps=10)

    try:
        query = "Use the add tool to calculate 42 + 58. Just give me the answer."
        logger.info("\n" + "=" * 80)
        logger.info("TEST: test_agent_run_with_reasoning")
        logger.info("=" * 80)
        logger.info(f"Query: {query}")

        # Capture stdout to verify plan was printed
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            result = await agent.run(query, reasoning=True)
        stdout_output = f.getvalue()

        logger.info(f"Result: {result}")
        logger.info(f"Tools used: {agent.tools_used_names}")
        logger.info("=" * 80 + "\n")

        # Verify reasoning plan was generated and printed
        assert (
            "REASONING PLAN" in stdout_output or "reasoning" in stdout_output.lower()
        ), "Reasoning plan should be printed"

        # Verify execution still works
        assert "100" in result
        assert len(agent.tools_used_names) > 0
        assert "add" in agent.tools_used_names

    finally:
        await agent.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_agent_stream_with_reasoning():
    """Test agent.stream() with reasoning=True yields plan first."""
    server_path = Path(__file__).parent.parent / "servers_for_testing" / "simple_server.py"

    config = {"mcpServers": {"simple": {"command": sys.executable, "args": [str(server_path), "--transport", "stdio"]}}}

    client = MCPClient.from_dict(config)
    llm = ChatOpenAI(model="gpt-4o")
    agent = MCPAgent(llm=llm, client=client, max_steps=5)

    try:
        query = "Add 10 and 20 using the add tool"
        logger.info("\n" + "=" * 80)
        logger.info("TEST: test_agent_stream_with_reasoning")
        logger.info("=" * 80)
        logger.info(f"Query: {query}")

        chunks = []
        plan_received = False
        intermediate_steps = []
        final_result = None

        async for chunk in agent.stream(query, reasoning=True):
            chunks.append(chunk)
            if isinstance(chunk, str):
                # Check if this is the reasoning plan
                if "REASONING PLAN" in chunk or "reasoning" in chunk.lower():
                    plan_received = True
                    logger.info(f"Reasoning plan received:\n{chunk[:500]}...")
                else:
                    final_result = chunk
                    logger.info(f"\nFinal result: {chunk}")

        logger.info(f"\nTotal chunks: {len(chunks)}")
        logger.info(f"Plan received: {plan_received}")
        logger.info(f"Intermediate steps: {len(intermediate_steps)}")
        logger.info(f"Tools used: {agent.tools_used_names}")
        logger.info("=" * 80 + "\n")

        # Verify reasoning plan was yielded first
        assert plan_received, "Reasoning plan should be yielded first"
        assert len(chunks) > 0, "Should yield at least one chunk"

        # Verify execution continued after plan
        assert final_result is not None, "Should have a final result"
        assert "30" in final_result, "Final result should contain the answer (30)"

        # Verify tools were used
        assert len(agent.tools_used_names) > 0, "Should have used at least one tool"
        assert "add" in agent.tools_used_names, "Should have used the 'add' tool"

    finally:
        await agent.close()

