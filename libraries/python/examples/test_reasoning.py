"""
Simple test script for testing the reasoning functionality.

This script demonstrates how to use the reasoning parameter with agent.run() and agent.stream().

Setup:
1. Create a .env file in the project root with your API keys:

   # Required: LLM API Key (choose one based on your LLM provider)
   OPENAI_API_KEY=your_openai_api_key_here
   # ANTHROPIC_API_KEY=your_anthropic_api_key_here
   # GOOGLE_API_KEY=your_google_api_key_here

   # Optional: Custom OpenAI base URL (for proxies or local deployments)
   # OPENAI_BASE_URL=https://api.openai.com/v1
   # Or use a proxy: OPENAI_BASE_URL=https://your-proxy.com/v1

   # Optional: Model name (can also be set in code)
   # OPENAI_MODEL=gpt-4o

   # Optional: MCP Use Remote Agent
   # MCP_USE_API_KEY=your_mcp_use_api_key

   # Optional: E2B Sandbox (for sandboxed execution)
   # E2B_API_KEY=your_e2b_api_key

   # Optional: Observability
   # LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
   # LANGFUSE_SECRET_KEY=your_langfuse_secret_key
   # LAMINAR_PROJECT_API_KEY=your_laminar_api_key

   # Optional: Debug mode
   # MCP_USE_DEBUG=true
   # DEBUG=true

   # Optional: Disable telemetry
   # MCP_USE_ANONYMIZED_TELEMETRY=false

2. Or export directly in your shell:
   export OPENAI_API_KEY=your_api_key_here

3. Install required packages:

   Option A: Install from source (recommended for development):
   cd libraries/python
   pip install -e ".[openai]"

   Option B: Install from PyPI:
   pip install python-dotenv langchain-openai mcp-use

4. Run the script:

   From libraries/python directory:
   python examples/test_reasoning.py

   Or from examples directory:
   cd libraries/python/examples
   python test_reasoning.py

Note: All environment variables can be configured in .env file. The script will
automatically load them using load_dotenv().
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path to allow importing mcp_use
# This allows running the script from the examples directory
script_dir = Path(__file__).parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dotenv import load_dotenv  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

from mcp_use import MCPAgent, MCPClient  # noqa: E402

# Load environment variables from .env file
# This will automatically load OPENAI_API_KEY if it's in the .env file
load_dotenv()

# Note: ChatOpenAI will automatically use OPENAI_API_KEY from environment variables
# You don't need to pass it explicitly when creating the LLM instance


async def test_reasoning_with_run():
    """Test reasoning with agent.run()."""
    print("\n" + "=" * 80)
    print("TEST: Reasoning with agent.run()")
    print("=" * 80)

    # Example: Using a simple MCP server
    # You can modify this config to use your own MCP servers
    config = {
        "mcpServers": {
            "everything": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-everything"],
            }
        }
    }

    client = MCPClient.from_dict(config)

    # Create LLM with customizable model and base_url from environment variables
    # Both model and base_url can be replaced:
    # - model: any OpenAI-compatible model name (e.g., "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo")
    # - base_url: custom API endpoint (for proxies, local deployments, or OpenAI-compatible services)
    model = os.getenv("OPENAI_MODEL", "gpt-4o")  # Default to gpt-4o if not set
    base_url = os.getenv("OPENAI_BASE_URL", None)  # Use default OpenAI API if not set

    llm_kwargs = {"model": model}
    if base_url:
        llm_kwargs["base_url"] = base_url
        print(f"Using custom base_url: {base_url}")

    print(f"Using model: {model}")
    llm = ChatOpenAI(**llm_kwargs)
    agent = MCPAgent(llm=llm, client=client, max_steps=10)

    try:
        query = "Find the best restaurant in San Francisco"
        print(f"\nQuery: {query}\n")

        # Run with reasoning=True
        print("Running with reasoning=True...")
        print("-" * 80)
        result = await agent.run(query, reasoning=True)
        print("-" * 80)

        print(f"\nFinal Result: {result}")
        print(f"Tools used: {agent.tools_used_names}")

        # Get reasoning plan if it was generated
        reasoning_plan = agent.get_reasoning_plan()
        if reasoning_plan:
            print(f"\n{'='*80}")
            print("Retrieved Reasoning Plan:")
            print(f"{'='*80}")
            print(reasoning_plan)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await agent.close()


async def test_reasoning_with_stream():
    """Test reasoning with agent.stream()."""
    print("\n" + "=" * 80)
    print("TEST: Reasoning with agent.stream()")
    print("=" * 80)

    config = {
        "mcpServers": {
            "everything": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-everything"],
            }
        }
    }

    client = MCPClient.from_dict(config)

    # Create LLM with customizable model and base_url from environment variables
    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    base_url = os.getenv("OPENAI_BASE_URL", None)

    llm_kwargs = {"model": model}
    if base_url:
        llm_kwargs["base_url"] = base_url
        print(f"Using custom base_url: {base_url}")

    print(f"Using model: {model}")
    llm = ChatOpenAI(**llm_kwargs)
    agent = MCPAgent(llm=llm, client=client, max_steps=10)

    try:
        query = "Find the best restaurant in San Francisco"
        print(f"\nQuery: {query}\n")

        # Stream with reasoning=True
        print("Streaming with reasoning=True...")
        print("-" * 80)

        plan_received = False
        async for chunk in agent.stream(query, reasoning=True):
            if isinstance(chunk, str):
                if "REASONING PLAN" in chunk or (not plan_received and len(chunk) > 100):
                    # This is likely the reasoning plan
                    print("\n[REASONING PLAN]")
                    print(chunk)
                    print("-" * 80)
                    plan_received = True
                else:
                    # This is the final result or intermediate output
                    print(chunk, end="", flush=True)
            elif isinstance(chunk, tuple):
                # Tool action and observation
                action, observation = chunk
                print(f"\n[Tool Call] {action.tool}: {action.tool_input}")
                print(f"[Result] {str(observation)[:200]}...")

        print("\n" + "-" * 80)
        print(f"\nTools used: {agent.tools_used_names}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await agent.close()


async def main():
    """Run all tests."""
    print("=" * 80)
    print("REASONING FUNCTIONALITY TEST")
    print("=" * 80)

    # Check if OpenAI API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("=" * 80)
        print("ERROR: OPENAI_API_KEY not found!")
        print("=" * 80)
        print("\nPlease set your OpenAI API key using one of the following methods:")
        print("\n1. Create a .env file in the project root with:")
        print("   OPENAI_API_KEY=your_api_key_here")
        print("\n2. Or export it in your shell:")
        print("   export OPENAI_API_KEY=your_api_key_here")
        print("\n3. Or set it inline (not recommended for production):")
        print("   os.environ['OPENAI_API_KEY'] = 'your_api_key_here'")
        print("\n" + "=" * 80)
        return

    print(f"✓ OpenAI API key found (starts with: {api_key[:10]}...)")

    try:
        # Test with run()
        await test_reasoning_with_run()

        # Test with stream()
        await test_reasoning_with_stream()

        print("\n" + "=" * 80)
        print("All tests completed!")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

