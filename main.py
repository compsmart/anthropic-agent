import sys
import asyncio
from config import AgentConfig
from agent import AnthropicAGIAgent


async def main():
    config = AgentConfig()
    if not config.api_key:
        import os
        if not os.getenv("ANTHROPIC_API_KEY"):
            print("❌ Error: ANTHROPIC_API_KEY environment variable not set")
            print("Please set your Anthropic API key:")
            print("export ANTHROPIC_API_KEY='your-api-key-here'")
            return
    agent = AnthropicAGIAgent(config)
    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
        response = await agent.process_message(user_input)
        print(response)
    else:
        await agent.run_interactive_mode()

if __name__ == "__main__":
    asyncio.run(main())
