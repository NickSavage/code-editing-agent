package main

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"agent/mocks"
)

func TestNewAgent(t *testing.T) {
	mockClient := &openai.Client{}
	mockInputManager := &InputManager{}
	model := "test-model"

	agent := NewAgent(mockClient, mockInputManager, model)

	assert.NotNil(t, agent)
	assert.Equal(t, mockClient, agent.client)
	assert.Equal(t, mockInputManager, agent.inputManager)
	assert.Equal(t, model, agent.model)
	assert.Len(t, agent.tools, 4) // read_file, list_dir, write_to_file, run_agent
	assert.Len(t, agent.toolHandlers, 4)
}

func TestAgent_SetupTools(t *testing.T) {
	agent := &Agent{
		toolHandlers: make(map[string]ToolHandler),
		model:        "test-model",
	}

	agent.setupTools()

	expectedTools := []string{"read_file", "list_dir", "write_to_file", "run_agent"}
	assert.Len(t, agent.tools, len(expectedTools))
	assert.Len(t, agent.toolHandlers, len(expectedTools))

	for _, toolName := range expectedTools {
		assert.Contains(t, agent.toolHandlers, toolName)
	}
}

func TestAgent_CreateErrorResponse(t *testing.T) {
	agent := setupTestAgent()

	toolCallID := "test-error-call"
	errorMsg := "test error message"

	response := agent.createErrorResponse(toolCallID, errorMsg)

	assert.Equal(t, openai.ChatMessageRoleTool, response.Role)
	assert.Equal(t, errorMsg, response.Content)
	assert.Equal(t, toolCallID, response.ToolCallID)
}

func TestAgent_CreateChatCompletion_Success(t *testing.T) {
	mockClient := mocks.NewMockOpenAIClient()
	agent := &Agent{
		client: mockClient,
		model:  "test-model",
		tools:  []openai.Tool{},
	}

	expectedResponse := mocks.CreateMockResponse("Hello, world!", nil)
	mockClient.AddResponse(expectedResponse)

	messages := []openai.ChatCompletionMessage{
		{
			Role:    openai.ChatMessageRoleUser,
			Content: "Hello",
		},
	}

	response, err := agent.createChatCompletion(context.Background(), messages)

	require.NoError(t, err)
	assert.Equal(t, "Hello, world!", response.Content)
	assert.Equal(t, openai.ChatMessageRoleAssistant, response.Role)
	assert.Equal(t, 1, mockClient.CallCount)
}

func TestAgent_CreateChatCompletion_Error(t *testing.T) {
	mockClient := mocks.NewMockOpenAIClient()
	agent := &Agent{
		client: mockClient,
		model:  "test-model",
		tools:  []openai.Tool{},
	}

	expectedError := assert.AnError
	mockClient.AddError(expectedError)

	messages := []openai.ChatCompletionMessage{
		{
			Role:    openai.ChatMessageRoleUser,
			Content: "Hello",
		},
	}

	_, err := agent.createChatCompletion(context.Background(), messages)

	require.Error(t, err)
	assert.Equal(t, expectedError, err)
	assert.Equal(t, 1, mockClient.CallCount)
}

func TestAgent_DriveConversation_NoToolCalls(t *testing.T) {
	mockClient := mocks.NewMockOpenAIClient()
	agent := &Agent{
		client:       mockClient,
		toolHandlers: make(map[string]ToolHandler),
		model:        "test-model",
		tools:        []openai.Tool{},
	}

	response := mocks.CreateMockResponse("Simple response without tool calls", nil)
	mockClient.AddResponse(response)

	initialMessages := []openai.ChatCompletionMessage{
		{
			Role:    openai.ChatMessageRoleUser,
			Content: "Hello",
		},
	}

	var logMessages []string
	logf := func(format string, args ...any) {
		logMessages = append(logMessages, format)
	}

	finalMessages, err := agent.DriveConversation(context.Background(), initialMessages, logf)

	require.NoError(t, err)
	assert.Len(t, finalMessages, 2) // Original user message + assistant response
	assert.Equal(t, "Simple response without tool calls", finalMessages[1].Content)
	assert.Equal(t, 1, mockClient.CallCount)
}

func TestAgent_DriveConversation_WithToolCalls(t *testing.T) {
	mockClient := mocks.NewMockOpenAIClient()
	agent := &Agent{
		client:       mockClient,
		toolHandlers: make(map[string]ToolHandler),
		model:        "test-model",
		tools:        []openai.Tool{},
	}
	agent.setupTools()

	// First response with tool call
	toolCall := mocks.CreateMockToolCall("call-1", "read_file", `{"path": "testdata/sample.txt"}`)
	firstResponse := mocks.CreateMockResponse("", []openai.ToolCall{toolCall})

	// Second response without tool calls (conversation ends)
	secondResponse := mocks.CreateMockResponse("File content processed", nil)

	mockClient.AddResponse(firstResponse)
	mockClient.AddResponse(secondResponse)

	initialMessages := []openai.ChatCompletionMessage{
		{
			Role:    openai.ChatMessageRoleUser,
			Content: "Read the sample file",
		},
	}

	var logMessages []string
	logf := func(format string, args ...any) {
		logMessages = append(logMessages, format)
	}

	finalMessages, err := agent.DriveConversation(context.Background(), initialMessages, logf)

	require.NoError(t, err)
	// Original user message + first assistant response + tool response + second assistant response
	assert.GreaterOrEqual(t, len(finalMessages), 4)
	assert.Equal(t, 2, mockClient.CallCount)
}

func TestAgent_DriveConversation_MaxIterations(t *testing.T) {
	mockClient := mocks.NewMockOpenAIClient()
	agent := &Agent{
		client:       mockClient,
		toolHandlers: make(map[string]ToolHandler),
		model:        "test-model",
		tools:        []openai.Tool{},
	}
	agent.setupTools()

	// Create responses that always have tool calls to trigger max iterations
	toolCall := mocks.CreateMockToolCall("call-1", "read_file", `{"path": "testdata/sample.txt"}`)

	// Add 11 responses (more than max iterations of 10)
	for i := 0; i < 11; i++ {
		response := mocks.CreateMockResponse("", []openai.ToolCall{toolCall})
		mockClient.AddResponse(response)
	}

	initialMessages := []openai.ChatCompletionMessage{
		{
			Role:    openai.ChatMessageRoleUser,
			Content: "Test max iterations",
		},
	}

	finalMessages, err := agent.DriveConversation(context.Background(), initialMessages, nil)

	require.NoError(t, err)
	// Should stop at max iterations (10), not process all 11 responses
	assert.Equal(t, 10, mockClient.CallCount)
	// Should have user message + 10 iterations of (assistant message + tool response)
	assert.GreaterOrEqual(t, len(finalMessages), 21)
}

func TestAgent_HandleRunAgent(t *testing.T) {
	mockClient := mocks.NewMockOpenAIClient()
	agent := &Agent{
		client:       mockClient,
		toolHandlers: make(map[string]ToolHandler),
		model:        "test-model",
		tools:        []openai.Tool{},
	}
	agent.setupTools()

	// Mock response for the sub-agent
	subAgentResponse := mocks.CreateMockResponse("Sub-agent completed the task", nil)
	mockClient.AddResponse(subAgentResponse)

	args, _ := json.Marshal(RunAgentInput{Task: "Test sub-agent task"})
	toolCall := openai.ToolCall{
		ID:   "run-agent-call",
		Type: "function",
		Function: openai.FunctionCall{
			Name:      "run_agent",
			Arguments: string(args),
		},
	}

	response := agent.handleRunAgent(toolCall)

	assert.Equal(t, openai.ChatMessageRoleTool, response.Role)
	assert.Contains(t, response.Content, "Agent task: Test sub-agent task")
	assert.Contains(t, response.Content, "Sub-agent completed the task")
	assert.Equal(t, "run-agent-call", response.ToolCallID)
	assert.Equal(t, 1, mockClient.CallCount)
}

func TestGetModel(t *testing.T) {
	// Test CLI argument takes priority
	cliModel := "cli-model"
	result := getModel(&cliModel)
	assert.Equal(t, "cli-model", result)

	// Test empty CLI argument falls back to env var
	emptyModel := ""
	t.Setenv("LLM_MODEL", "env-model")
	result = getModel(&emptyModel)
	assert.Equal(t, "env-model", result)

	// Test nil CLI argument falls back to env var
	t.Setenv("LLM_MODEL", "env-model-2")
	result = getModel(nil)
	assert.Equal(t, "env-model-2", result)

	// Test falls back to default when no env var
	t.Setenv("LLM_MODEL", "")
	result = getModel(nil)
	assert.Equal(t, DEFAULT_MODEL, result)
}