package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func setupTestAgent() *Agent {
	return &Agent{
		toolHandlers: make(map[string]ToolHandler),
		model:        "test-model",
	}
}

func TestHandleReadFile_Success(t *testing.T) {
	agent := setupTestAgent()
	agent.setupTools()

	// Create test file
	testContent := "test file content"
	testFile := filepath.Join("testdata", "test_read.txt")
	err := os.WriteFile(testFile, []byte(testContent), 0644)
	require.NoError(t, err)
	defer os.Remove(testFile)

	// Create tool call
	args, _ := json.Marshal(ReadFileInput{Path: testFile})
	toolCall := openai.ToolCall{
		ID:   "test-call-1",
		Type: "function",
		Function: openai.FunctionCall{
			Name:      "read_file",
			Arguments: string(args),
		},
	}

	response := agent.handleReadFile(toolCall)

	assert.Equal(t, openai.ChatMessageRoleTool, response.Role)
	assert.Equal(t, testContent, response.Content)
	assert.Equal(t, "test-call-1", response.ToolCallID)
}

func TestHandleReadFile_FileNotFound(t *testing.T) {
	agent := setupTestAgent()
	agent.setupTools()

	args, _ := json.Marshal(ReadFileInput{Path: "nonexistent.txt"})
	toolCall := openai.ToolCall{
		ID:   "test-call-2",
		Type: "function",
		Function: openai.FunctionCall{
			Name:      "read_file",
			Arguments: string(args),
		},
	}

	response := agent.handleReadFile(toolCall)

	assert.Equal(t, openai.ChatMessageRoleTool, response.Role)
	assert.Contains(t, response.Content, "Error reading file")
	assert.Equal(t, "test-call-2", response.ToolCallID)
}

func TestHandleReadFile_InvalidJSON(t *testing.T) {
	agent := setupTestAgent()
	agent.setupTools()

	toolCall := openai.ToolCall{
		ID:   "test-call-3",
		Type: "function",
		Function: openai.FunctionCall{
			Name:      "read_file",
			Arguments: "invalid json",
		},
	}

	response := agent.handleReadFile(toolCall)

	assert.Equal(t, openai.ChatMessageRoleTool, response.Role)
	assert.Contains(t, response.Content, "Invalid arguments")
	assert.Equal(t, "test-call-3", response.ToolCallID)
}

func TestHandleListDir_Success(t *testing.T) {
	agent := setupTestAgent()
	agent.setupTools()

	args, _ := json.Marshal(ListDirInput{Path: "testdata", Recursive: false})
	toolCall := openai.ToolCall{
		ID:   "test-call-4",
		Type: "function",
		Function: openai.FunctionCall{
			Name:      "list_dir",
			Arguments: string(args),
		},
	}

	response := agent.handleListDir(toolCall)

	assert.Equal(t, openai.ChatMessageRoleTool, response.Role)
	assert.Contains(t, response.Content, "sample.txt")
	assert.Contains(t, response.Content, "empty_file.txt")
	assert.Equal(t, "test-call-4", response.ToolCallID)
}

func TestHandleListDir_Recursive(t *testing.T) {
	agent := setupTestAgent()
	agent.setupTools()

	args, _ := json.Marshal(ListDirInput{Path: "testdata", Recursive: true})
	toolCall := openai.ToolCall{
		ID:   "test-call-5",
		Type: "function",
		Function: openai.FunctionCall{
			Name:      "list_dir",
			Arguments: string(args),
		},
	}

	response := agent.handleListDir(toolCall)

	assert.Equal(t, openai.ChatMessageRoleTool, response.Role)
	assert.Contains(t, response.Content, "test_dir")
	assert.Contains(t, response.Content, "test_dir/nested_file.txt")
	assert.Equal(t, "test-call-5", response.ToolCallID)
}

func TestHandleListDir_DirectoryNotFound(t *testing.T) {
	agent := setupTestAgent()
	agent.setupTools()

	args, _ := json.Marshal(ListDirInput{Path: "nonexistent_dir", Recursive: false})
	toolCall := openai.ToolCall{
		ID:   "test-call-6",
		Type: "function",
		Function: openai.FunctionCall{
			Name:      "list_dir",
			Arguments: string(args),
		},
	}

	response := agent.handleListDir(toolCall)

	assert.Equal(t, openai.ChatMessageRoleTool, response.Role)
	assert.Contains(t, response.Content, "Error reading directory")
	assert.Equal(t, "test-call-6", response.ToolCallID)
}

func TestHandleWriteFile_Success(t *testing.T) {
	agent := setupTestAgent()
	agent.setupTools()

	testFile := filepath.Join("testdata", "test_write.txt")
	testContent := "written content"
	defer os.Remove(testFile)

	args, _ := json.Marshal(WriteFileInput{Path: testFile, Content: testContent})
	toolCall := openai.ToolCall{
		ID:   "test-call-7",
		Type: "function",
		Function: openai.FunctionCall{
			Name:      "write_to_file",
			Arguments: string(args),
		},
	}

	response := agent.handleWriteFile(toolCall)

	assert.Equal(t, openai.ChatMessageRoleTool, response.Role)
	assert.Equal(t, "File written successfully.", response.Content)
	assert.Equal(t, "test-call-7", response.ToolCallID)

	// Verify file was written
	content, err := os.ReadFile(testFile)
	require.NoError(t, err)
	assert.Equal(t, testContent, string(content))
}

func TestHandleWriteFile_CreateDirectory(t *testing.T) {
	agent := setupTestAgent()
	agent.setupTools()

	testDir := filepath.Join("testdata", "new_dir")
	testFile := filepath.Join(testDir, "test_file.txt")
	testContent := "content in new directory"
	defer os.RemoveAll(testDir)

	args, _ := json.Marshal(WriteFileInput{Path: testFile, Content: testContent})
	toolCall := openai.ToolCall{
		ID:   "test-call-8",
		Type: "function",
		Function: openai.FunctionCall{
			Name:      "write_to_file",
			Arguments: string(args),
		},
	}

	response := agent.handleWriteFile(toolCall)

	assert.Equal(t, openai.ChatMessageRoleTool, response.Role)
	assert.Equal(t, "File written successfully.", response.Content)
	assert.Equal(t, "test-call-8", response.ToolCallID)

	// Verify directory and file were created
	content, err := os.ReadFile(testFile)
	require.NoError(t, err)
	assert.Equal(t, testContent, string(content))
}

func TestProcessToolCalls_MultipleTools(t *testing.T) {
	agent := setupTestAgent()
	agent.setupTools()

	// Create test file for reading
	testFile := filepath.Join("testdata", "multi_test.txt")
	testContent := "multi tool test"
	err := os.WriteFile(testFile, []byte(testContent), 0644)
	require.NoError(t, err)
	defer os.Remove(testFile)

	readArgs, _ := json.Marshal(ReadFileInput{Path: testFile})
	listArgs, _ := json.Marshal(ListDirInput{Path: "testdata", Recursive: false})

	toolCalls := []openai.ToolCall{
		{
			ID:   "call-1",
			Type: "function",
			Function: openai.FunctionCall{
				Name:      "read_file",
				Arguments: string(readArgs),
			},
		},
		{
			ID:   "call-2",
			Type: "function",
			Function: openai.FunctionCall{
				Name:      "list_dir",
				Arguments: string(listArgs),
			},
		},
	}

	responses := agent.processToolCalls(toolCalls)

	require.Len(t, responses, 2)
	assert.Equal(t, testContent, responses[0].Content)
	assert.Contains(t, responses[1].Content, "sample.txt")
}

func TestProcessToolCalls_UnknownTool(t *testing.T) {
	agent := setupTestAgent()
	agent.setupTools()

	toolCalls := []openai.ToolCall{
		{
			ID:   "call-unknown",
			Type: "function",
			Function: openai.FunctionCall{
				Name:      "unknown_tool",
				Arguments: "{}",
			},
		},
	}

	responses := agent.processToolCalls(toolCalls)

	require.Len(t, responses, 1)
	assert.Contains(t, responses[0].Content, "Unknown tool")
	assert.Equal(t, "call-unknown", responses[0].ToolCallID)
}