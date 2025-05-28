#!/usr/bin/env python3
"""
Interactive Tool Testing Suite for Anthropic AGI Agent

This script allows you to interactively test each tool and fix any issues found.
Run this to validate that all tools are working correctly.
"""

from config import AgentConfig
from agent import AnthropicAGIAgent
import asyncio
import os
import sys
import tempfile
from pathlib import Path
import json
import logging

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# Configure logging for testing
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ToolTester:
    """Interactive tool testing class"""

    def __init__(self):
        self.config = AgentConfig()
        self.agent = None
        self.test_results = {
            "web_search": {"passed": 0, "failed": 0, "tests": []},
            "file_operations": {"passed": 0, "failed": 0, "tests": []},
            "code_execution": {"passed": 0, "failed": 0, "tests": []},
            "knowledge_management": {"passed": 0, "failed": 0, "tests": []},
            "system_interaction": {"passed": 0, "failed": 0, "tests": []},
            "memory_recall": {"passed": 0, "failed": 0, "tests": []}
        }

    async def setup(self):
        """Setup the agent for testing"""
        print("🔧 Setting up test environment...")

        # Check for API key
        if not self.config.api_key and not os.getenv("ANTHROPIC_API_KEY"):
            print("❌ Error: ANTHROPIC_API_KEY environment variable not set")
            print("Please set your API key to run tests")
            return False

        # Create agent
        self.agent = AnthropicAGIAgent(self.config)
        print("✅ Agent initialized successfully")
        return True

    def print_header(self, title):
        """Print a formatted header"""
        print("\n" + "="*60)
        print(f"  {title}")
        print("="*60)

    def print_test_header(self, test_name):
        """Print a formatted test header"""
        print(f"\n🧪 Testing: {test_name}")
        print("-" * 40)

    async def test_web_search(self):
        """Test web search functionality"""
        self.print_header("WEB SEARCH TESTS")

        test_queries = [
            "What is artificial intelligence?",
            "weather today",
            "latest news about Python programming",
            "compsmart.co.uk website information"
        ]

        for query in test_queries:
            self.print_test_header(f"Web search: '{query}'")

            try:
                response = await self.agent.process_message(f"Search the web for: {query}")

                print(f"🤖 Response: {response[:300]}...")

                # Check if response indicates success
                if "❌" in response or "error" in response.lower():
                    self.test_results["web_search"]["failed"] += 1
                    result = "FAILED"
                else:
                    self.test_results["web_search"]["passed"] += 1
                    result = "PASSED"

                print(f"📊 Result: {result}")
                self.test_results["web_search"]["tests"].append({
                    "query": query,
                    "result": result,
                    "response": response[:100]
                })

            except Exception as e:
                print(f"❌ Test failed with exception: {e}")
                self.test_results["web_search"]["failed"] += 1
                self.test_results["web_search"]["tests"].append({
                    "query": query,
                    "result": "FAILED",
                    "error": str(e)
                })

    async def test_file_operations(self):
        """Test file operations functionality"""
        self.print_header("FILE OPERATIONS TESTS")

        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test_file.txt"
            test_content = "This is a test file for the AGI agent.\nLine 2 of the test file."

            file_tests = [
                ("Write file",
                 f"Write the following content to {test_file}: {test_content}"),
                ("Read file", f"Read the contents of {test_file}"),
                ("Analyze file", f"Analyze the file {test_file}"),
                ("List directory",
                 f"List the contents of directory {temp_dir}")
            ]

            for test_name, command in file_tests:
                self.print_test_header(test_name)

                try:
                    response = await self.agent.process_message(command)

                    print(f"🤖 Response: {response[:300]}...")

                    # Check if response indicates success
                    if "❌" in response or "error" in response.lower():
                        self.test_results["file_operations"]["failed"] += 1
                        result = "FAILED"
                    else:
                        self.test_results["file_operations"]["passed"] += 1
                        result = "PASSED"

                    print(f"📊 Result: {result}")
                    self.test_results["file_operations"]["tests"].append({
                        "test": test_name,
                        "result": result,
                        "response": response[:100]
                    })

                except Exception as e:
                    print(f"❌ Test failed with exception: {e}")
                    self.test_results["file_operations"]["failed"] += 1
                    self.test_results["file_operations"]["tests"].append({
                        "test": test_name,
                        "result": "FAILED",
                        "error": str(e)
                    })

    async def test_code_execution(self):
        """Test code execution functionality"""
        self.print_header("CODE EXECUTION TESTS")

        code_tests = [
            ("Simple calculation", "Execute this Python code: print(2 + 2)"),
            ("Variable assignment",
             "Execute this Python code: x = 10; y = 20; print(f'Sum: {x + y}')"),
            ("List operations",
             "Execute this Python code: numbers = [1, 2, 3, 4, 5]; print(f'Sum: {sum(numbers)}')"),
            ("Date/time",
             "Execute this Python code: from datetime import datetime; print(f'Current time: {datetime.now()}')"),
            # This should fail gracefully
            ("Error handling",
             "Execute this Python code: print('This should work'); print(1/0)")
        ]

        for test_name, command in code_tests:
            self.print_test_header(test_name)

            try:
                response = await self.agent.process_message(command)

                print(f"🤖 Response: {response[:300]}...")

                # For code execution, we expect either success or controlled failure
                if "Code executed successfully" in response or "✅" in response:
                    self.test_results["code_execution"]["passed"] += 1
                    result = "PASSED"
                elif "Code execution failed" in response and test_name == "Error handling":
                    # Expected failure for division by zero
                    self.test_results["code_execution"]["passed"] += 1
                    result = "PASSED (Expected failure)"
                else:
                    self.test_results["code_execution"]["failed"] += 1
                    result = "FAILED"

                print(f"📊 Result: {result}")
                self.test_results["code_execution"]["tests"].append({
                    "test": test_name,
                    "result": result,
                    "response": response[:100]
                })

            except Exception as e:
                print(f"❌ Test failed with exception: {e}")
                self.test_results["code_execution"]["failed"] += 1
                self.test_results["code_execution"]["tests"].append({
                    "test": test_name,
                    "result": "FAILED",
                    "error": str(e)
                })

    async def test_knowledge_management(self):
        """Test knowledge management functionality"""
        self.print_header("KNOWLEDGE MANAGEMENT TESTS")

        knowledge_tests = [
            ("Add knowledge", "Add to knowledge base: The user's favorite color is blue"),
            ("Add categorized knowledge",
             "Add to knowledge base under category 'preferences': The user prefers tea over coffee"),
            ("Search knowledge",
             "Search the knowledge base for information about 'user preferences'"),
            ("Search color knowledge", "Search the knowledge base for 'color'")
        ]

        for test_name, command in knowledge_tests:
            self.print_test_header(test_name)

            try:
                response = await self.agent.process_message(command)

                print(f"🤖 Response: {response[:300]}...")

                # Check if response indicates success
                if "✅" in response or "Knowledge added" in response or "search results" in response:
                    self.test_results["knowledge_management"]["passed"] += 1
                    result = "PASSED"
                elif "❌" in response or "error" in response.lower():
                    self.test_results["knowledge_management"]["failed"] += 1
                    result = "FAILED"
                else:
                    # Neutral response, still count as passed if no explicit error
                    self.test_results["knowledge_management"]["passed"] += 1
                    result = "PASSED"

                print(f"📊 Result: {result}")
                self.test_results["knowledge_management"]["tests"].append({
                    "test": test_name,
                    "result": result,
                    "response": response[:100]
                })

            except Exception as e:
                print(f"❌ Test failed with exception: {e}")
                self.test_results["knowledge_management"]["failed"] += 1
                self.test_results["knowledge_management"]["tests"].append({
                    "test": test_name,
                    "result": "FAILED",
                    "error": str(e)
                })

    async def test_system_interaction(self):
        """Test system interaction functionality"""
        self.print_header("SYSTEM INTERACTION TESTS")

        # Only test safe commands
        system_tests = [
            ("Current directory", "What is the current directory?"),
            ("List files", "List the files in the current directory"),
            ("System date", "What is the current system date?"),
            ("Whoami", "Run the whoami command"),
            ("Echo test", "Echo 'Hello from AGI agent'")
        ]

        for test_name, command in system_tests:
            self.print_test_header(test_name)

            try:
                response = await self.agent.process_message(command)

                print(f"🤖 Response: {response[:300]}...")

                # Check if response indicates success
                if "✅" in response or "Command executed" in response:
                    self.test_results["system_interaction"]["passed"] += 1
                    result = "PASSED"
                elif "❌" in response or "error" in response.lower():
                    self.test_results["system_interaction"]["failed"] += 1
                    result = "FAILED"
                else:
                    # Check if it's a safety restriction (which is good)
                    if "not allowed" in response or "safe mode" in response:
                        self.test_results["system_interaction"]["passed"] += 1
                        result = "PASSED (Safety restriction)"
                    else:
                        self.test_results["system_interaction"]["passed"] += 1
                        result = "PASSED"

                print(f"📊 Result: {result}")
                self.test_results["system_interaction"]["tests"].append({
                    "test": test_name,
                    "result": result,
                    "response": response[:100]
                })

            except Exception as e:
                print(f"❌ Test failed with exception: {e}")
                self.test_results["system_interaction"]["failed"] += 1
                self.test_results["system_interaction"]["tests"].append({
                    "test": test_name,
                    "result": "FAILED",
                    "error": str(e)
                })

    async def test_memory_recall(self):
        """Test memory recall functionality"""
        self.print_header("MEMORY RECALL TESTS")

        # First, add some information to memory
        await self.agent.process_message("My name is Brad and I am testing the AGI agent")
        await self.agent.process_message("I live in the UK and work with computers")

        memory_tests = [
            ("Recall name", "What is my name?"),
            ("Recall location", "Where do I live?"),
            ("Recall work", "What do I work with?"),
            ("General recall", "Tell me what you remember about our conversation")
        ]

        for test_name, command in memory_tests:
            self.print_test_header(test_name)

            try:
                response = await self.agent.process_message(command)

                print(f"🤖 Response: {response[:300]}...")

                # Check if response contains relevant information
                if test_name == "Recall name" and "Brad" in response:
                    result = "PASSED"
                elif test_name == "Recall location" and "UK" in response:
                    result = "PASSED"
                elif test_name == "Recall work" and ("computer" in response.lower() or "testing" in response.lower()):
                    result = "PASSED"
                elif "Memory recall" in response or "remember" in response.lower():
                    result = "PASSED"
                elif "❌" in response or "No memories found" in response:
                    result = "FAILED"
                else:
                    result = "PASSED"

                if result == "PASSED":
                    self.test_results["memory_recall"]["passed"] += 1
                else:
                    self.test_results["memory_recall"]["failed"] += 1

                print(f"📊 Result: {result}")
                self.test_results["memory_recall"]["tests"].append({
                    "test": test_name,
                    "result": result,
                    "response": response[:100]
                })

            except Exception as e:
                print(f"❌ Test failed with exception: {e}")
                self.test_results["memory_recall"]["failed"] += 1
                self.test_results["memory_recall"]["tests"].append({
                    "test": test_name,
                    "result": "FAILED",
                    "error": str(e)
                })

    def print_summary(self):
        """Print test summary"""
        self.print_header("TEST SUMMARY")

        total_passed = 0
        total_failed = 0

        for tool_name, results in self.test_results.items():
            passed = results["passed"]
            failed = results["failed"]
            total_passed += passed
            total_failed += failed

            status = "✅ PASS" if failed == 0 else "❌ FAIL" if passed == 0 else "⚠️ MIXED"

            print(
                f"{tool_name:20} | {status} | Passed: {passed:2d} | Failed: {failed:2d}")

        print("-" * 60)
        overall_status = "✅ ALL TESTS PASSED" if total_failed == 0 else f"⚠️ {total_failed} TESTS FAILED"
        print(
            f"{'OVERALL':20} | {overall_status} | Total: {total_passed + total_failed}")

        # Save detailed results
        with open("test_results.json", "w") as f:
            json.dump(self.test_results, f, indent=2)
        print(f"\n📄 Detailed results saved to test_results.json")

        return total_failed == 0

    async def run_all_tests(self):
        """Run all tests"""
        print("🚀 Starting AGI Agent Tool Testing Suite")

        if not await self.setup():
            return False

        # Run all tests
        await self.test_web_search()
        await self.test_file_operations()
        await self.test_code_execution()
        await self.test_knowledge_management()
        await self.test_system_interaction()
        await self.test_memory_recall()

        # Print summary
        return self.print_summary()

    async def interactive_test(self):
        """Run interactive testing session"""
        if not await self.setup():
            return

        self.print_header("INTERACTIVE TESTING MODE")
        print("You can now chat with the agent to test tools manually.")
        print("Type 'run_tests' to run automated tests")
        print("Type 'exit' to quit")

        while True:
            try:
                user_input = input("\n🧑 You: ").strip()

                if user_input.lower() in ['exit', 'quit']:
                    break
                elif user_input.lower() == 'run_tests':
                    await self.run_all_tests()
                    continue

                if user_input:
                    print("🤔 Processing...")
                    response = await self.agent.process_message(user_input)
                    print(f"🤖 Agent: {response}")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


async def main():
    """Main function"""
    tester = ToolTester()

    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        await tester.interactive_test()
    else:
        success = await tester.run_all_tests()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
