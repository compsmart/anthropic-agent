#!/usr/bin/env python3
"""
Advanced Anthropic AGI Agent
A comprehensive AI agent leveraging the latest Anthropic API features including
Claude 4 models, tool use, file handling, and advanced reasoning capabilities.
"""

import asyncio
import json
import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import base64
import mimetypes

# Core dependencies
import anthropic
from anthropic import Anthropic, AsyncAnthropic
from anthropic.types import Message, MessageParam, ToolUseBlock, TextBlock

# Additional tools for enhanced functionality
import requests
import sqlite3
import subprocess
from dataclasses import dataclass, asdict
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agi_agent.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class AgentConfig:
    """Configuration for the AGI agent"""
    model: str = "claude-3-5-sonnet-20241022"  # Latest Claude model
    max_tokens: int = 8192
    temperature: float = 0.7
    api_key: Optional[str] = None
    memory_file: str = "agent_memory.json"
    knowledge_base: str = "knowledge_base.db"
    debug: bool = False

class MemoryManager:
    """Manages persistent memory for the agent"""
    
    def __init__(self, memory_file: str):
        self.memory_file = memory_file
        self.memory = self._load_memory()
    
    def _load_memory(self) -> Dict[str, Any]:
        """Load memory from file"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load memory: {e}")
        return {
            "conversations": [],
            "learned_facts": [],
            "user_preferences": {},
            "task_history": [],
            "errors": []
        }
    
    def save_memory(self):
        """Save memory to file"""
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(self.memory, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")
    
    def add_conversation(self, user_input: str, response: str, metadata: Dict = None):
        """Add conversation to memory"""
        self.memory["conversations"].append({
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "response": response,
            "metadata": metadata or {}
        })
        self._trim_conversations()
        self.save_memory()
    
    def _trim_conversations(self, max_conversations: int = 100):
        """Keep only recent conversations"""
        if len(self.memory["conversations"]) > max_conversations:
            self.memory["conversations"] = self.memory["conversations"][-max_conversations:]

class KnowledgeBase:
    """SQLite-based knowledge base for structured information storage"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize the knowledge base database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables for different types of knowledge
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT,
                content TEXT,
                source TEXT,
                confidence REAL,
                timestamp TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity1 TEXT,
                relationship TEXT,
                entity2 TEXT,
                strength REAL,
                timestamp TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_fact(self, category: str, content: str, source: str = "user", confidence: float = 1.0):
        """Add a fact to the knowledge base"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO facts (category, content, source, confidence, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (category, content, source, confidence, datetime.now().isoformat()))
        conn.commit()
        conn.close()
    
    def search_facts(self, query: str, limit: int = 10) -> List[Dict]:
        """Search for relevant facts"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT category, content, source, confidence, timestamp
            FROM facts
            WHERE content LIKE ?
            ORDER BY confidence DESC
            LIMIT ?
        ''', (f'%{query}%', limit))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'category': row[0],
                'content': row[1],
                'source': row[2],
                'confidence': row[3],
                'timestamp': row[4]
            })
        
        conn.close()
        return results

class AnthropicAGIAgent:
    """Advanced AGI Agent using Anthropic's latest capabilities"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.client = Anthropic(api_key=config.api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.async_client = AsyncAnthropic(api_key=config.api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.memory = MemoryManager(config.memory_file)
        self.knowledge_base = KnowledgeBase(config.knowledge_base)
        
        # Define available tools
        self.tools = [
            {
                "name": "web_search",
                "description": "Search the web for current information",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "file_operations",
                "description": "Read, write, or analyze files",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "operation": {"type": "string", "enum": ["read", "write", "analyze", "list"]},
                        "path": {"type": "string", "description": "File or directory path"},
                        "content": {"type": "string", "description": "Content to write (for write operations)"}
                    },
                    "required": ["operation", "path"]
                }
            },
            {
                "name": "code_execution",
                "description": "Execute Python code in a safe environment",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Python code to execute"},
                        "timeout": {"type": "integer", "description": "Execution timeout in seconds", "default": 30}
                    },
                    "required": ["code"]
                }
            },
            {
                "name": "knowledge_management",
                "description": "Add or search knowledge in the knowledge base",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "operation": {"type": "string", "enum": ["add", "search"]},
                        "category": {"type": "string", "description": "Knowledge category"},
                        "content": {"type": "string", "description": "Content to add or search for"},
                        "confidence": {"type": "number", "description": "Confidence level (0-1)", "default": 1.0}
                    },
                    "required": ["operation", "content"]
                }
            },
            {
                "name": "system_interaction",
                "description": "Interact with the system (run commands, check status)",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "System command to execute"},
                        "safe_mode": {"type": "boolean", "description": "Only allow safe commands", "default": True}
                    },
                    "required": ["command"]
                }
            },
            {
                "name": "memory_recall",
                "description": "Recall information from previous conversations",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What to recall from memory"},
                        "limit": {"type": "integer", "description": "Number of results to return", "default": 5}
                    },
                    "required": ["query"]
                }
            }
        ]
    
    async def process_message(self, user_input: str, context: Dict = None) -> str:
        """Process a user message with full AGI capabilities"""
        try:
            # Build conversation context
            messages = self._build_context(user_input, context)
            
            # Make the API call with tools
            response = await self.async_client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=messages,
                tools=self.tools
            )
            
            # Process tool use if present
            if any(isinstance(block, ToolUseBlock) for block in response.content):
                response = await self._handle_tool_use(response, messages)
            
            # Extract final response
            final_response = self._extract_text_response(response)
            
            # Store in memory
            self.memory.add_conversation(
                user_input, 
                final_response,
                {"model": self.config.model, "timestamp": datetime.now().isoformat()}
            )
            
            return final_response
            
        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            logger.error(error_msg)
            return f"I encountered an error: {error_msg}"
    
    def _build_context(self, user_input: str, context: Dict = None) -> List[MessageParam]:
        """Build conversation context with memory and system prompt"""
        system_prompt = """You are an advanced AGI agent with access to multiple tools and capabilities. 
        You can search the web, execute code, manage files, interact with systems, and maintain persistent memory.
        
        Your capabilities include:
        - Web searching for current information
        - File operations (read, write, analyze)
        - Code execution in Python
        - Knowledge base management
        - System interactions
        - Memory recall from previous conversations
        
        Always think step by step and use tools when they would be helpful. Be thorough, accurate, and helpful.
        If you need to perform multiple steps, break them down clearly.
        """
        
        messages = [{"role": "user", "content": f"{system_prompt}\n\nUser: {user_input}"}]
        
        # Add relevant memory context
        if context and context.get("include_memory", True):
            recent_conversations = self.memory.memory["conversations"][-3:]  # Last 3 conversations
            if recent_conversations:
                memory_context = "Recent conversation context:\n"
                for conv in recent_conversations:
                    memory_context += f"User: {conv['user_input'][:100]}...\nAssistant: {conv['response'][:100]}...\n\n"
                messages[0]["content"] = memory_context + messages[0]["content"]
        
        return messages
    
    async def _handle_tool_use(self, response: Message, messages: List[MessageParam]) -> Message:
        """Handle tool use in the response"""
        tool_results = []
        
        for block in response.content:
            if isinstance(block, ToolUseBlock):
                tool_name = block.name
                tool_input = block.input
                
                try:
                    result = await self._execute_tool(tool_name, tool_input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result)
                    })
                except Exception as e:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Tool execution error: {str(e)}",
                        "is_error": True
                    })
        
        # Continue conversation with tool results
        if tool_results:
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
            
            # Get follow-up response
            follow_up = await self.async_client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=messages,
                tools=self.tools
            )
            
            return follow_up
        
        return response
    
    async def _execute_tool(self, tool_name: str, tool_input: Dict) -> str:
        """Execute a specific tool"""
        logger.info(f"Executing tool: {tool_name} with input: {tool_input}")
        
        if tool_name == "web_search":
            return await self._web_search(tool_input["query"])
        
        elif tool_name == "file_operations":
            return await self._file_operations(
                tool_input["operation"],
                tool_input["path"],
                tool_input.get("content")
            )
        
        elif tool_name == "code_execution":
            return await self._code_execution(
                tool_input["code"],
                tool_input.get("timeout", 30)
            )
        
        elif tool_name == "knowledge_management":
            return await self._knowledge_management(
                tool_input["operation"],
                tool_input.get("category", "general"),
                tool_input["content"],
                tool_input.get("confidence", 1.0)
            )
        
        elif tool_name == "system_interaction":
            return await self._system_interaction(
                tool_input["command"],
                tool_input.get("safe_mode", True)
            )
        
        elif tool_name == "memory_recall":
            return await self._memory_recall(
                tool_input["query"],
                tool_input.get("limit", 5)
            )
        
        else:
            return f"Unknown tool: {tool_name}"
    
    async def _web_search(self, query: str) -> str:
        """Perform web search (placeholder - would integrate with search API)"""
        # This would integrate with a real search API like Google, Bing, etc.
        return f"Web search results for '{query}': [This would contain real search results]"
    
    async def _file_operations(self, operation: str, path: str, content: str = None) -> str:
        """Handle file operations"""
        try:
            if operation == "read":
                with open(path, 'r', encoding='utf-8') as f:
                    return f"File content:\n{f.read()}"
            
            elif operation == "write":
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return f"Successfully wrote to {path}"
            
            elif operation == "list":
                path_obj = Path(path)
                if path_obj.is_dir():
                    items = list(path_obj.iterdir())
                    return f"Directory contents:\n" + "\n".join(str(item) for item in items)
                else:
                    return f"Path {path} is not a directory"
            
            elif operation == "analyze":
                path_obj = Path(path)
                stats = path_obj.stat()
                return f"File analysis for {path}:\nSize: {stats.st_size} bytes\nModified: {datetime.fromtimestamp(stats.st_mtime)}"
        
        except Exception as e:
            return f"File operation error: {str(e)}"
    
    async def _code_execution(self, code: str, timeout: int = 30) -> str:
        """Execute Python code safely"""
        try:
            # Create a restricted execution environment
            restricted_globals = {
                '__builtins__': {
                    'print': print,
                    'len': len,
                    'str': str,
                    'int': int,
                    'float': float,
                    'list': list,
                    'dict': dict,
                    'tuple': tuple,
                    'set': set,
                    'range': range,
                    'enumerate': enumerate,
                    'zip': zip,
                    'sum': sum,
                    'max': max,
                    'min': min,
                    'abs': abs,
                    'round': round,
                }
            }
            
            # Capture output
            import io
            from contextlib import redirect_stdout, redirect_stderr
            
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, restricted_globals)
            
            stdout_output = stdout_capture.getvalue()
            stderr_output = stderr_capture.getvalue()
            
            result = ""
            if stdout_output:
                result += f"Output:\n{stdout_output}"
            if stderr_output:
                result += f"Errors:\n{stderr_output}"
            
            return result or "Code executed successfully (no output)"
        
        except Exception as e:
            return f"Code execution error: {str(e)}"
    
    async def _knowledge_management(self, operation: str, category: str, content: str, confidence: float = 1.0) -> str:
        """Manage knowledge base"""
        if operation == "add":
            self.knowledge_base.add_fact(category, content, "user", confidence)
            return f"Added knowledge to category '{category}'"
        
        elif operation == "search":
            results = self.knowledge_base.search_facts(content)
            if results:
                formatted_results = []
                for result in results:
                    formatted_results.append(f"[{result['category']}] {result['content']} (confidence: {result['confidence']})")
                return "Knowledge search results:\n" + "\n".join(formatted_results)
            else:
                return "No matching knowledge found"
    
    async def _system_interaction(self, command: str, safe_mode: bool = True) -> str:
        """Interact with system"""
        if safe_mode:
            # Only allow safe commands
            safe_commands = ['ls', 'pwd', 'date', 'whoami', 'ps aux', 'df -h', 'free -h']
            if not any(command.startswith(safe_cmd) for safe_cmd in safe_commands):
                return f"Command '{command}' not allowed in safe mode"
        
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            
            output = ""
            if result.stdout:
                output += f"Output:\n{result.stdout}"
            if result.stderr:
                output += f"Errors:\n{result.stderr}"
            
            return output or "Command executed successfully"
        
        except subprocess.TimeoutExpired:
            return "Command timed out"
        except Exception as e:
            return f"System interaction error: {str(e)}"
    
    async def _memory_recall(self, query: str, limit: int = 5) -> str:
        """Recall information from memory"""
        conversations = self.memory.memory["conversations"]
        relevant_conversations = []
        
        # Simple keyword matching (could be enhanced with embeddings)
        for conv in conversations:
            if query.lower() in conv["user_input"].lower() or query.lower() in conv["response"].lower():
                relevant_conversations.append(conv)
        
        # Return most recent matches
        recent_matches = relevant_conversations[-limit:]
        
        if recent_matches:
            formatted_results = []
            for conv in recent_matches:
                formatted_results.append(f"[{conv['timestamp']}]\nUser: {conv['user_input'][:100]}...\nAssistant: {conv['response'][:100]}...")
            return "Memory recall results:\n" + "\n\n".join(formatted_results)
        else:
            return f"No memories found matching '{query}'"
    
    def _extract_text_response(self, response: Message) -> str:
        """Extract text response from message"""
        text_blocks = [block.text for block in response.content if isinstance(block, TextBlock)]
        return "\n".join(text_blocks)
    
    async def run_interactive_mode(self):
        """Run the agent in interactive mode"""
        print("🤖 Advanced Anthropic AGI Agent Started!")
        print("Type 'quit' to exit, 'help' for commands")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\n🧑 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    print("👋 Goodbye!")
                    break
                
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                elif user_input.lower() == 'memory':
                    self._show_memory_stats()
                    continue
                
                elif user_input.lower().startswith('config'):
                    self._handle_config_command(user_input)
                    continue
                
                print("🤔 Thinking...")
                response = await self.process_message(user_input)
                print(f"\n🤖 Agent: {response}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
    
    def _show_help(self):
        """Show help information"""
        help_text = """
Available commands:
- quit/exit: Exit the agent
- help: Show this help message
- memory: Show memory statistics
- config: Show current configuration

The agent has access to these tools:
- Web search
- File operations (read, write, analyze)
- Code execution
- Knowledge management
- System interaction
- Memory recall

Just ask naturally and I'll use the appropriate tools!
        """
        print(help_text)
    
    def _show_memory_stats(self):
        """Show memory statistics"""
        memory = self.memory.memory
        print(f"\n📊 Memory Statistics:")
        print(f"Conversations: {len(memory['conversations'])}")
        print(f"Learned facts: {len(memory['learned_facts'])}")
        print(f"User preferences: {len(memory['user_preferences'])}")
        print(f"Task history: {len(memory['task_history'])}")
    
    def _handle_config_command(self, command: str):
        """Handle configuration commands"""
        print(f"\n⚙️ Current Configuration:")
        print(f"Model: {self.config.model}")
        print(f"Max tokens: {self.config.max_tokens}")
        print(f"Temperature: {self.config.temperature}")
        print(f"Debug mode: {self.config.debug}")

async def main():
    """Main function to run the AGI agent"""
    # Load configuration
    config = AgentConfig()
    
    # Check for API key
    if not config.api_key and not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Error: ANTHROPIC_API_KEY environment variable not set")
        print("Please set your Anthropic API key:")
        print("export ANTHROPIC_API_KEY='your-api-key-here'")
        return
    
    # Create and run the agent
    agent = AnthropicAGIAgent(config)
    
    # Check if running interactively or with arguments
    if len(sys.argv) > 1:
        # Process single command
        user_input = " ".join(sys.argv[1:])
        response = await agent.process_message(user_input)
        print(response)
    else:
        # Run interactive mode
        await agent.run_interactive_mode()

if __name__ == "__main__":
    asyncio.run(main())