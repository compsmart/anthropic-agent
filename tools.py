import subprocess
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, Any, List
import json
import sys
import os

logger = logging.getLogger(__name__)


class Toolset:
    """Defines and executes tools for the agent."""

    def __init__(self, memory, knowledge_base):
        self.memory = memory
        self.knowledge_base = knowledge_base

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Return the tool definitions for the Anthropic API."""
        return [
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

    async def execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """Execute a tool and return the result."""
        try:
            if tool_name == "web_search":
                return await self._web_search(tool_input)
            elif tool_name == "file_operations":
                return await self._file_operations(tool_input)
            elif tool_name == "code_execution":
                return await self._code_execution(tool_input)
            elif tool_name == "knowledge_management":
                return await self._knowledge_management(tool_input)
            elif tool_name == "system_interaction":
                return await self._system_interaction(tool_input)
            elif tool_name == "memory_recall":
                return await self._memory_recall(tool_input)
            else:
                return f"❌ Unknown tool: {tool_name}"
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return f"❌ Error executing {tool_name}: {str(e)}"

    async def _web_search(self, tool_input: Dict[str, Any]) -> str:
        """Execute web search using requests and basic HTML parsing."""
        query = tool_input.get("query", "")

        try:
            # Use DuckDuckGo Instant Answer API for basic search
            search_url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1"
            }

            import requests
            response = requests.get(search_url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()

                # Extract relevant information
                result = f"🔍 Web search results for '{query}':\n\n"

                # Abstract (direct answer)
                if data.get("Abstract"):
                    result += f"Summary: {data['Abstract']}\n\n"

                # Answer (instant answer)
                if data.get("Answer"):
                    result += f"Answer: {data['Answer']}\n\n"

                # Related topics
                if data.get("RelatedTopics"):
                    result += "Related information:\n"
                    for i, topic in enumerate(data["RelatedTopics"][:3]):
                        if isinstance(topic, dict) and "Text" in topic:
                            result += f"• {topic['Text'][:200]}...\n"

                # Definition if available
                if data.get("Definition"):
                    result += f"\nDefinition: {data['Definition']}\n"

                # If no useful content found
                if not any([data.get("Abstract"), data.get("Answer"), data.get("RelatedTopics"), data.get("Definition")]):
                    result += "No direct results found. You may want to search more specifically or try a different search engine."

                return result
            else:
                return f"❌ Web search failed with status code: {response.status_code}"

        except requests.exceptions.Timeout:
            return "❌ Web search timed out. Please try again."
        except requests.exceptions.RequestException as e:
            return f"❌ Web search failed: {str(e)}"
        except Exception as e:
            return f"❌ Web search error: {str(e)}"

    async def _file_operations(self, tool_input: Dict[str, Any]) -> str:
        """Execute file operations."""
        operation = tool_input.get("operation")
        path = tool_input.get("path")
        content = tool_input.get("content")

        try:
            file_path = Path(path)

            if operation == "read":
                if file_path.exists() and file_path.is_file():
                    return file_path.read_text(encoding='utf-8')
                else:
                    return f"❌ File not found: {path}"

            elif operation == "write":
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content or "", encoding='utf-8')
                return f"✅ File written successfully: {path}"

            elif operation == "list":
                if file_path.exists() and file_path.is_dir():
                    items = [str(item) for item in file_path.iterdir()]
                    return f"📁 Contents of {path}:\n" + "\n".join(items)
                else:
                    return f"❌ Directory not found: {path}"

            elif operation == "analyze":
                if file_path.exists() and file_path.is_file():
                    stats = file_path.stat()
                    return f"📊 File analysis for {path}:\nSize: {stats.st_size} bytes\nModified: {datetime.fromtimestamp(stats.st_mtime)}"
                else:
                    return f"❌ File not found: {path}"

            else:
                return f"❌ Unknown operation: {operation}"

        except Exception as e:
            return f"❌ File operation error: {str(e)}"

    async def _code_execution(self, tool_input: Dict[str, Any]) -> str:
        """Execute Python code safely."""
        code = tool_input.get("code", "")
        timeout = tool_input.get("timeout", 30)

        try:
            # Create a subprocess to execute the code safely
            process = subprocess.Popen(
                [sys.executable, "-c", code],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout
            )
            stdout, stderr = process.communicate(timeout=timeout)

            if process.returncode == 0:
                return f"✅ Code executed successfully:\n{stdout}"
            else:
                return f"❌ Code execution failed:\n{stderr}"

        except subprocess.TimeoutExpired:
            return f"❌ Code execution timed out after {timeout} seconds"
        except Exception as e:
            return f"❌ Code execution error: {str(e)}"

    async def _knowledge_management(self, tool_input: Dict[str, Any]) -> str:
        """Manage knowledge base operations."""
        operation = tool_input.get("operation")
        category = tool_input.get("category", "general")
        content = tool_input.get("content", "")
        confidence = tool_input.get("confidence", 1.0)

        try:
            if operation == "add":
                # Add to both knowledge base and memory
                self.knowledge_base.add_knowledge(
                    category, content, confidence)
                self.memory.add_learned_fact(content, category, confidence)
                return f"✅ Knowledge added to category '{category}' and stored in memory"
            elif operation == "search":
                results = self.knowledge_base.search(content)
                if results:
                    return f"🔍 Knowledge search results:\n" + "\n".join(results)
                else:
                    return "❌ No knowledge found matching the search query"
            else:
                return f"❌ Unknown knowledge operation: {operation}"
        except Exception as e:
            return f"❌ Knowledge management error: {str(e)}"

    async def _system_interaction(self, tool_input: Dict[str, Any]) -> str:
        """Execute system commands safely."""
        command = tool_input.get("command", "")
        safe_mode = tool_input.get("safe_mode", True)

        # Define safe commands
        safe_commands = [
            "ls", "dir", "pwd", "whoami", "date", "time", "echo",
            "cat", "head", "tail", "grep", "find", "which", "where"
        ]

        if safe_mode:
            command_parts = command.split()
            if not command_parts or command_parts[0] not in safe_commands:
                return f"❌ Command '{command}' not allowed in safe mode"

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                return f"✅ Command executed:\n{result.stdout}"
            else:
                return f"❌ Command failed:\n{result.stderr}"

        except subprocess.TimeoutExpired:
            return "❌ Command timed out"
        except Exception as e:
            return f"❌ System interaction error: {str(e)}"

    async def _memory_recall(self, tool_input: Dict[str, Any]) -> str:
        """Recall information from memory."""
        query = tool_input.get("query", "")
        limit = tool_input.get("limit", 5)

        try:
            # Search both memory and learned facts
            memory_results = self.memory.search(query, limit)

            if memory_results:
                return f"🧠 Memory recall results:\n" + "\n".join(memory_results)
            else:
                return "❌ No memories found matching the query"
        except Exception as e:
            return f"❌ Memory recall error: {str(e)}"
