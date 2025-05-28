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

# Import custom modules
from config import AgentConfig
from memory import MemoryManager
from knowledge_base import KnowledgeBase
from tools import Toolset

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


class AnthropicAGIAgent:
    """Advanced AGI Agent using Anthropic's latest capabilities"""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.client = Anthropic(
            api_key=config.api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.async_client = AsyncAnthropic(
            api_key=config.api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.memory = MemoryManager(config.memory_file)
        self.knowledge_base = KnowledgeBase(config.knowledge_base)

        # Initialize toolset and get available tools
        self.toolset = Toolset(self.memory, self.knowledge_base)
        self.tools = self.toolset.get_tool_definitions()

        # Conversation state
        self.conversation_messages = []
        self.conversation_started = False

    async def process_message(self, user_input: str, context: Dict = None) -> str:
        """Process a user message with full AGI capabilities"""
        try:
            # Build messages for this request
            messages = self._build_messages(user_input, context)

            # Get response from Claude
            api_params = {
                "model": self.config.model,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "messages": messages,
                "tools": self.tools
            }

            # Add system prompt only for the first message
            if not self.conversation_started:
                api_params["system"] = self._build_system_prompt()
                self.conversation_started = True

            response = await self.async_client.messages.create(**api_params)

            # Handle tool use if present
            if response.content and any(hasattr(block, 'type') and block.type == "tool_use" for block in response.content):
                response = await self._handle_tool_use(response, messages)

            # Extract final response text
            final_response = ""
            if response.content:
                for content in response.content:
                    if hasattr(content, 'text'):
                        final_response += content.text

            # Update conversation history
            self.conversation_messages.append(
                {"role": "user", "content": user_input})
            self.conversation_messages.append(
                {"role": "assistant", "content": final_response})

            # Save conversation to memory
            self.memory.add_conversation(
                user_input=user_input,
                response=final_response,
                metadata={
                    "model": self.config.model,
                    "timestamp": datetime.now().isoformat()
                }
            )

            return final_response

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            error_msg = f"❌ I encountered an error: {str(e)}"

            # Still save the conversation even if there was an error
            self.memory.add_conversation(
                user_input=user_input,
                response=error_msg,
                metadata={
                    "error": True,
                    "timestamp": datetime.now().isoformat()
                }
            )

            return error_msg

    def _build_system_prompt(self) -> str:
        """Build the system prompt with current context"""
        system_content = self.config.system_message

        # Add current time
        system_content += f"\n\nCurrent time: {datetime.now().isoformat()}"

        # Add recent memory context for first message
        recent_conversations = self.memory.get_recent_conversations(3)
        if recent_conversations:
            system_content += "\n\nRecent conversation context:\n"
            for conv in recent_conversations:
                system_content += f"User: {conv['user_input']}\nAgent: {conv['response'][:200]}...\n\n"

        return system_content

    def _build_messages(self, user_input: str, context: Dict = None) -> List[MessageParam]:
        """Build messages list for API call"""
        # Start with existing conversation history
        messages = self.conversation_messages.copy()

        # Add the current user input
        messages.append({"role": "user", "content": user_input})

        return messages

    async def _handle_tool_use(self, response: Message, messages: List[MessageParam]) -> Message:
        """Handle tool use in the response"""
        tool_results = []

        for block in response.content:
            if isinstance(block, ToolUseBlock):
                tool_name = block.name
                tool_input = block.input

                logger.info(
                    f"Executing tool: {tool_name} with input: {tool_input}")

                try:
                    result = await self.toolset.execute_tool(tool_name, tool_input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result)
                    })
                except Exception as e:
                    logger.error(f"Tool execution error: {e}")
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

    def reset_conversation(self):
        """Reset the conversation state"""
        self.conversation_messages = []
        self.conversation_started = False

    async def _system_interaction(self, command: str, safe_mode: bool = True) -> str:
        """Interact with system"""
        if safe_mode:
            # Only allow safe commands
            safe_commands = ['ls', 'pwd', 'date',
                             'whoami', 'ps aux', 'df -h', 'free -h']
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
                formatted_results.append(
                    f"[{conv['timestamp']}]\nUser: {conv['user_input'][:100]}...\nAssistant: {conv['response'][:100]}...")
            return "Memory recall results:\n" + "\n\n".join(formatted_results)
        else:
            return f"No memories found matching '{query}'"

    def _extract_text_response(self, response: Message) -> str:
        """Extract text response from message"""
        text_blocks = [
            block.text for block in response.content if isinstance(block, TextBlock)]
        return "\n".join(text_blocks)

    async def run_interactive_mode(self):
        """Run the agent in interactive mode"""
        print("🤖 Advanced Anthropic AGI Agent")
        print("Type 'exit', 'quit', or 'bye' to end the conversation")
        print("Type 'help' for available commands")
        print("Type 'memory' to see memory statistics")
        print("Type 'reset' to start a new conversation")
        print("-" * 50)

        while True:
            try:
                user_input = input("\n🧑 You: ").strip()

                if not user_input:
                    continue

                # Handle special commands
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("\n👋 Goodbye!")
                    break
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                elif user_input.lower() == 'memory':
                    self._show_memory_stats()
                    continue
                elif user_input.lower() == 'reset':
                    self.reset_conversation()
                    print("\n🔄 Conversation reset. Starting fresh!")
                    continue
                elif user_input.lower().startswith('config'):
                    self._handle_config_command(user_input)
                    continue

                # Process the message
                print("🤔 Thinking...")
                response = await self.process_message(user_input)
                print(f"\n🤖 Agent: {response}")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")
                print(f"\n❌ Error: {e}")

    def _show_help(self):
        """Show help information"""
        help_text = """
🔧 Available Commands:
- exit, quit, bye: End the conversation
- help: Show this help message
- memory: Show memory statistics
- reset: Start a new conversation
- config: Show current configuration

🛠️ Available Tools:
1. Web search - Search for current information
2. File operations - Read, write, analyze files
3. Code execution - Run Python code safely
4. Knowledge management - Store and retrieve knowledge
5. System interaction - Execute system commands
6. Memory recall - Recall previous conversations

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
        print(f"Temperature: {self.config.temperature}")
        print(f"Debug mode: {self.config.debug}")
