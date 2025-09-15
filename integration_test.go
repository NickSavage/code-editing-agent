package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"agent/mocks"
)

func TestIntegration_ReadWriteWorkflow(t *testing.T) {
	mockClient := mocks.NewMockOpenAIClient()
	agent := NewAgent(mockClient, nil, "test-model")

	// Setup test scenario: AI wants to read a file, then write a new file
	readArgs, _ := json.Marshal(ReadFileInput{Path: "testdata/sample.txt"})
	writeArgs, _ := json.Marshal(WriteFileInput{
		Path:    "testdata/integration_output.txt",
		Content: "Processed content from sample.txt",
	})

	// First response: AI wants to read file
	readToolCall := mocks.CreateMockToolCall("call-1", "read_file", string(readArgs))
	firstResponse := mocks.CreateMockResponse("I'll read the file first", []openai.ToolCall{readToolCall})

	// Second response: AI wants to write file based on what it read
	writeToolCall := mocks.CreateMockToolCall("call-2", "write_to_file", string(writeArgs))
	secondResponse := mocks.CreateMockResponse("Now I'll write the processed content", []openai.ToolCall{writeToolCall})

	// Third response: AI is done
	thirdResponse := mocks.CreateMockResponse("Task completed successfully", nil)

	mockClient.AddResponse(firstResponse)
	mockClient.AddResponse(secondResponse)
	mockClient.AddResponse(thirdResponse)

	// Start conversation
	initialMessages := []openai.ChatCompletionMessage{
		{
			Role:    openai.ChatMessageRoleUser,
			Content: "Read the sample file and create a processed version",
		},
	}

	var logOutput []string
	logf := func(format string, args ...any) {
		logOutput = append(logOutput, format)
	}

	finalMessages, err := agent.DriveConversation(context.Background(), initialMessages, logf)

	require.NoError(t, err)
	assert.Equal(t, 3, mockClient.CallCount)

	// Verify the output file was created
	outputFile := "testdata/integration_output.txt"
	defer os.Remove(outputFile)

	content, err := os.ReadFile(outputFile)
	require.NoError(t, err)
	assert.Equal(t, "Processed content from sample.txt", string(content))

	// Verify conversation flow
	assert.GreaterOrEqual(t, len(finalMessages), 6) // user + 3*(assistant + tool response)

	// Check that tools were used by looking for the expected format strings
	foundReadFile := false
	foundWriteFile := false
	for _, log := range logOutput {
		if log == "Tool used: %s" {
			foundReadFile = true
		}
		if log == "Tool used: %s" {
			foundWriteFile = true
		}
	}
	assert.True(t, foundReadFile || foundWriteFile, "Should have tool usage logs")
}

func TestIntegration_DirectoryExplorationWorkflow(t *testing.T) {
	mockClient := mocks.NewMockOpenAIClient()
	agent := NewAgent(mockClient, nil, "test-model")

	// Create a temporary directory structure for testing
	tempDir := filepath.Join("testdata", "temp_explore")
	err := os.MkdirAll(filepath.Join(tempDir, "subdir"), 0755)
	require.NoError(t, err)
	defer os.RemoveAll(tempDir)

	err = os.WriteFile(filepath.Join(tempDir, "file1.txt"), []byte("content1"), 0644)
	require.NoError(t, err)
	err = os.WriteFile(filepath.Join(tempDir, "subdir", "file2.txt"), []byte("content2"), 0644)
	require.NoError(t, err)

	// Setup conversation: list directory, then recursively
	listArgs, _ := json.Marshal(ListDirInput{Path: tempDir, Recursive: false})
	recursiveListArgs, _ := json.Marshal(ListDirInput{Path: tempDir, Recursive: true})

	// First response: AI wants to list directory
	listToolCall := mocks.CreateMockToolCall("call-1", "list_dir", string(listArgs))
	firstResponse := mocks.CreateMockResponse("Let me explore the directory", []openai.ToolCall{listToolCall})

	// Second response: AI wants recursive listing
	recursiveListToolCall := mocks.CreateMockToolCall("call-2", "list_dir", string(recursiveListArgs))
	secondResponse := mocks.CreateMockResponse("Now let me see the full structure", []openai.ToolCall{recursiveListToolCall})

	// Third response: AI is done
	thirdResponse := mocks.CreateMockResponse("Directory exploration complete", nil)

	mockClient.AddResponse(firstResponse)
	mockClient.AddResponse(secondResponse)
	mockClient.AddResponse(thirdResponse)

	initialMessages := []openai.ChatCompletionMessage{
		{
			Role:    openai.ChatMessageRoleUser,
			Content: "Explore the temp directory structure",
		},
	}

	finalMessages, err := agent.DriveConversation(context.Background(), initialMessages, nil)

	require.NoError(t, err)
	assert.Equal(t, 3, mockClient.CallCount)
	assert.GreaterOrEqual(t, len(finalMessages), 6)

	// Verify tool responses contain expected file listings
	var toolResponses []string
	for _, msg := range finalMessages {
		if msg.Role == openai.ChatMessageRoleTool {
			toolResponses = append(toolResponses, msg.Content)
		}
	}

	assert.Len(t, toolResponses, 2)
	assert.Contains(t, toolResponses[0], "file1.txt")
	assert.Contains(t, toolResponses[1], "subdir/file2.txt")
}

func TestIntegration_ErrorHandling(t *testing.T) {
	mockClient := mocks.NewMockOpenAIClient()
	agent := NewAgent(mockClient, nil, "test-model")

	// Setup scenario with file that doesn't exist
	readArgs, _ := json.Marshal(ReadFileInput{Path: "nonexistent_file.txt"})

	// AI tries to read non-existent file
	readToolCall := mocks.CreateMockToolCall("call-1", "read_file", string(readArgs))
	firstResponse := mocks.CreateMockResponse("Let me read that file", []openai.ToolCall{readToolCall})

	// AI handles the error gracefully
	secondResponse := mocks.CreateMockResponse("I see the file doesn't exist", nil)

	mockClient.AddResponse(firstResponse)
	mockClient.AddResponse(secondResponse)

	initialMessages := []openai.ChatCompletionMessage{
		{
			Role:    openai.ChatMessageRoleUser,
			Content: "Read the nonexistent file",
		},
	}

	finalMessages, err := agent.DriveConversation(context.Background(), initialMessages, nil)

	require.NoError(t, err)
	assert.Equal(t, 2, mockClient.CallCount)

	// Find the tool error response
	var errorResponse string
	for _, msg := range finalMessages {
		if msg.Role == openai.ChatMessageRoleTool && msg.ToolCallID == "call-1" {
			errorResponse = msg.Content
			break
		}
	}

	assert.Contains(t, errorResponse, "Error reading file")
}

func TestIntegration_SubAgentExecution(t *testing.T) {
	mockClient := mocks.NewMockOpenAIClient()
	agent := NewAgent(mockClient, nil, "test-model")

	// Main agent wants to run a sub-agent
	runAgentArgs, _ := json.Marshal(RunAgentInput{Task: "Create a test file with specific content"})

	// Main agent response: run sub-agent
	runAgentToolCall := mocks.CreateMockToolCall("call-1", "run_agent", string(runAgentArgs))
	mainResponse := mocks.CreateMockResponse("I'll use a sub-agent for this task", []openai.ToolCall{runAgentToolCall})

	// Sub-agent response: create file
	writeArgs, _ := json.Marshal(WriteFileInput{
		Path:    "testdata/subagent_output.txt",
		Content: "Content created by sub-agent",
	})
	writeToolCall := mocks.CreateMockToolCall("sub-call-1", "write_to_file", string(writeArgs))
	subAgentResponse := mocks.CreateMockResponse("Creating the requested file", []openai.ToolCall{writeToolCall})

	// Sub-agent completion
	subAgentComplete := mocks.CreateMockResponse("File created successfully", nil)

	// Main agent final response
	mainComplete := mocks.CreateMockResponse("Sub-agent task completed", nil)

	mockClient.AddResponse(mainResponse)
	mockClient.AddResponse(subAgentResponse)
	mockClient.AddResponse(subAgentComplete)
	mockClient.AddResponse(mainComplete)

	initialMessages := []openai.ChatCompletionMessage{
		{
			Role:    openai.ChatMessageRoleUser,
			Content: "Use a sub-agent to create a test file",
		},
	}

	finalMessages, err := agent.DriveConversation(context.Background(), initialMessages, nil)

	require.NoError(t, err)

	// Cleanup
	defer os.Remove("testdata/subagent_output.txt")

	// Verify file was created by sub-agent
	content, err := os.ReadFile("testdata/subagent_output.txt")
	require.NoError(t, err)
	assert.Equal(t, "Content created by sub-agent", string(content))

	// Verify run_agent tool response contains sub-agent output
	var runAgentResponse string
	for _, msg := range finalMessages {
		if msg.Role == openai.ChatMessageRoleTool && msg.ToolCallID == "call-1" {
			runAgentResponse = msg.Content
			break
		}
	}

	assert.Contains(t, runAgentResponse, "Agent task: Create a test file with specific content")
	assert.Contains(t, runAgentResponse, "Creating the requested file")
}

func TestIntegration_ComplexMultiToolWorkflow(t *testing.T) {
	mockClient := mocks.NewMockOpenAIClient()
	agent := NewAgent(mockClient, nil, "test-model")

	// Complex workflow: list dir -> read files -> process -> write summary
	listArgs, _ := json.Marshal(ListDirInput{Path: "testdata", Recursive: false})
	readArgs, _ := json.Marshal(ReadFileInput{Path: "testdata/sample.txt"})
	writeArgs, _ := json.Marshal(WriteFileInput{
		Path:    "testdata/summary.txt",
		Content: "Summary: Found sample.txt with test content",
	})

	// Response 1: List directory
	listToolCall := mocks.CreateMockToolCall("call-1", "list_dir", string(listArgs))
	response1 := mocks.CreateMockResponse("Let me explore the directory", []openai.ToolCall{listToolCall})

	// Response 2: Read interesting file
	readToolCall := mocks.CreateMockToolCall("call-2", "read_file", string(readArgs))
	response2 := mocks.CreateMockResponse("I found sample.txt, let me read it", []openai.ToolCall{readToolCall})

	// Response 3: Write summary
	writeToolCall := mocks.CreateMockToolCall("call-3", "write_to_file", string(writeArgs))
	response3 := mocks.CreateMockResponse("Now I'll write a summary", []openai.ToolCall{writeToolCall})

	// Response 4: Complete
	response4 := mocks.CreateMockResponse("Analysis complete", nil)

	mockClient.AddResponse(response1)
	mockClient.AddResponse(response2)
	mockClient.AddResponse(response3)
	mockClient.AddResponse(response4)

	initialMessages := []openai.ChatCompletionMessage{
		{
			Role:    openai.ChatMessageRoleUser,
			Content: "Analyze the testdata directory and create a summary",
		},
	}

	finalMessages, err := agent.DriveConversation(context.Background(), initialMessages, nil)

	require.NoError(t, err)
	assert.Equal(t, 4, mockClient.CallCount)

	// Cleanup
	defer os.Remove("testdata/summary.txt")

	// Verify summary file was created
	content, err := os.ReadFile("testdata/summary.txt")
	require.NoError(t, err)
	assert.Equal(t, "Summary: Found sample.txt with test content", string(content))

	// Verify conversation structure
	assert.GreaterOrEqual(t, len(finalMessages), 8) // user + 4*(assistant + tool response)
}