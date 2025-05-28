# Anthropic AGI Agent

A comprehensive AI agent leveraging the latest Anthropic API features including Claude models, tool use, file handling, and advanced reasoning capabilities.

## Project Overview

This agent is designed to showcase the capabilities of Anthropic's Claude models in an AGI (Artificial General Intelligence) context, with a focus on:

- Context-aware conversation handling
- Multi-tool integration
- Persistent memory
- Knowledge base management
- System interactions
- File operations

## Features

- **Web Search**: Access to current information from the web
- **File Operations**: Read, write, and analyze files
- **Code Execution**: Safe Python code execution
- **Knowledge Management**: Structured storage of facts and relationships
- **System Interaction**: Execute system commands
- **Memory Recall**: Access information from previous conversations

## Getting Started

### Prerequisites

- Python 3.8+
- An Anthropic API key

### Installation

1. Clone the repository
2. Set up a virtual environment:
   ```
   python -m venv venv
   ```
3. Activate the virtual environment:
   ```
   # On Windows
   .\venv\Scripts\activate
   
   # On Mac/Linux
   source venv/bin/activate
   ```
4. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

### Configuration

Set your Anthropic API key as an environment variable:

```
# On Windows
set ANTHROPIC_API_KEY=your-api-key-here

# On Mac/Linux
export ANTHROPIC_API_KEY=your-api-key-here
```

Alternatively, you can provide the API key directly in the `AgentConfig` class.

## Usage

```python
from agent import AnthropicAGIAgent, AgentConfig

# Initialize with default configuration
config = AgentConfig()
agent = AnthropicAGIAgent(config)

# Process a message
response = await agent.process_message("Tell me about the latest AI developments")
print(response)
```

## Project Structure

- `agent.py`: Main agent implementation
- `requirements.txt`: Required Python packages
- `agent_memory.json`: Persistent agent memory (created on first run)
- `knowledge_base.db`: SQLite database for storing structured knowledge

## Requirements

- anthropic
- requests
- pyyaml

## License

[MIT License](LICENSE)

## Acknowledgements

- [Anthropic Claude API](https://anthropic.com)
- Contributors to the project
