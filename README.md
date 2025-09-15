# AI Agent with File System Tools

Experiment with a basic AI coding assistant. The agent uses OpenAI's function calling API to provide file operations like reading, writing, and listing directories. It's only really been tested with OpenRouter though.

## Features

- **Interactive Chat Interface**: Command-line interface for chatting with the AI
- **File System Tools**: Built-in tools for file operations:
  - `read_file`: Read contents of files
  - `list_dir`: List directory contents (with optional recursive listing)
  - `write_to_file`: Write content to files (creates directories as needed)
- **Configurable LLM Backend**: Works with any OpenAI-compatible API endpoint
- **Tool Calling**: Seamless integration between AI responses and tool execution

## Prerequisites

- Go 1.22.5 or later
- Access to an OpenAI-compatible API endpoint

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd agent
   ```

2. Install dependencies:
   ```bash
   go mod tidy
   ```

3. Set up environment variables in a `.env` file:
   ```bash
   LLM_ENDPOINT=your-api-endpoint
   LLM_KEY=your-api-key
   ```

## Usage

1. Run the agent:
   ```bash
   go run main.go
   ```

2. Start chatting with the AI assistant. The agent can help you with:
   - Reading and analyzing files
   - Creating and modifying files
   - Exploring directory structures
   - General coding and development tasks

3. Use `Ctrl+C` to quit the application

## Configuration

The agent is configured via environment variables:

- `LLM_ENDPOINT`: The base URL for your LLM API endpoint (required)
- `LLM_KEY`: Your API key for authentication

The default model is set to `anthropic/claude-sonnet-4` but can be changed by modifying the `MODEL` constant in `main.go`.

## File System Tools

### read_file
Reads the contents of a file at the specified path.
- **Input**: `path` (string) - Relative path to the file
- **Output**: File contents as text

### list_dir
Lists the contents of a directory.
- **Input**: 
  - `path` (string) - Relative path to the directory
  - `recursive` (boolean) - Whether to list recursively
- **Output**: List of files and directories

### write_to_file
Writes content to a file, creating directories as needed.
- **Input**:
  - `path` (string) - Relative path where to write the file
  - `content` (string) - Content to write
- **Output**: Success confirmation

## Project Structure

```
.
├── main.go           # Main application code
├── go.mod            # Go module definition
├── go.sum            # Go module checksums
├── .env              # Environment variables (not tracked)
├── .gitignore        # Git ignore rules
└── README.md         # This file
```

## Dependencies

- `github.com/sashabaranov/go-openai v1.41.2` - OpenAI API client for Go

## Development

The agent is built with a modular structure:

- **Agent struct**: Main agent with tool capabilities
- **Tool definitions**: OpenAI function calling schemas
- **Tool handlers**: Implementation of each tool's functionality
- **Chat loop**: Interactive conversation management

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]