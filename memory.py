import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class MemoryManager:
    """Manages persistent memory for the agent"""

    def __init__(self, memory_file: str):
        self.memory_file = memory_file
        self.memory = self._load_memory()

    def _load_memory(self) -> Dict[str, Any]:
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading memory: {e}")
                return self._create_empty_memory()
        return self._create_empty_memory()

    def _create_empty_memory(self) -> Dict[str, Any]:
        """Create empty memory structure"""
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
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.memory, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving memory: {e}")

    def add_conversation(self, user_input: str, response: str, metadata: Dict = None):
        """Add a conversation to memory"""
        conversation = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "response": response,
            "metadata": metadata or {}
        }
        self.memory["conversations"].append(conversation)
        self.save_memory()

    def search(self, query: str, limit: int = 5) -> List[str]:
        """Search through conversations and facts for relevant information"""
        results = []
        query_lower = query.lower()

        # Search conversations
        # Most recent first
        for conv in reversed(self.memory["conversations"]):
            if len(results) >= limit:
                break

            # Check user input and response for keywords
            if (query_lower in conv["user_input"].lower() or
                    query_lower in conv["response"].lower()):

                result = f"[{conv['timestamp'][:19]}] User: {conv['user_input'][:100]}..."
                if len(conv['response']) > 0:
                    result += f"\nAgent: {conv['response'][:100]}..."
                results.append(result)

        # Search learned facts
        for fact in self.memory["learned_facts"]:
            if len(results) >= limit:
                break
            if query_lower in fact.get("content", "").lower():
                results.append(f"Fact: {fact['content']}")

        return results

    def add_learned_fact(self, fact: str, category: str = "general", confidence: float = 1.0):
        """Add a learned fact to memory"""
        fact_entry = {
            "content": fact,
            "category": category,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        }
        self.memory["learned_facts"].append(fact_entry)
        self.save_memory()

    def get_recent_conversations(self, limit: int = 5) -> List[Dict]:
        """Get recent conversations"""
        return self.memory["conversations"][-limit:] if self.memory["conversations"] else []

    def _trim_conversations(self, max_conversations: int = 100):
        """Keep only recent conversations"""
        if len(self.memory["conversations"]) > max_conversations:
            self.memory["conversations"] = self.memory["conversations"][-max_conversations:]
            self.save_memory()
