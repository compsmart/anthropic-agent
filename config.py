from dataclasses import dataclass
from typing import Optional


@dataclass
class AgentConfig:
    """Configuration for the AGI agent"""
    model: str = "claude-3-5-sonnet-20241022"
    max_tokens: int = 4000
    temperature: float = 0.7
    api_key: Optional[str] = None
    memory_file: str = "agent_memory.json"
    knowledge_base: str = "knowledge_base.db"
    debug: bool = False
    system_message: str = """You are an advanced AI agent with access to multiple tools and capabilities. You can:

1. Search the web for current information
2. Read, write, and analyze files 
3. Execute Python code safely
4. Manage a knowledge base
5. Perform system interactions
6. Recall information from previous conversations

Use tools when appropriate to help the user accomplish their goals. Be thorough and helpful.
Always explain what you're doing when using tools.

You are running on a windows system and speaking to a person named Brad.
"""
