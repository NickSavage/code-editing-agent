package mocks

import (
	"context"
	"errors"

	"github.com/sashabaranov/go-openai"
)

type MockOpenAIClient struct {
	ChatCompletionResponses []openai.ChatCompletionResponse
	ChatCompletionErrors    []error
	CallCount               int
}

func NewMockOpenAIClient() *MockOpenAIClient {
	return &MockOpenAIClient{
		ChatCompletionResponses: make([]openai.ChatCompletionResponse, 0),
		ChatCompletionErrors:    make([]error, 0),
		CallCount:               0,
	}
}

func (m *MockOpenAIClient) CreateChatCompletion(ctx context.Context, request openai.ChatCompletionRequest) (openai.ChatCompletionResponse, error) {
	if m.CallCount >= len(m.ChatCompletionResponses) {
		if m.CallCount < len(m.ChatCompletionErrors) {
			err := m.ChatCompletionErrors[m.CallCount]
			m.CallCount++
			return openai.ChatCompletionResponse{}, err
		}
		return openai.ChatCompletionResponse{}, errors.New("no more mock responses configured")
	}

	response := m.ChatCompletionResponses[m.CallCount]
	m.CallCount++
	return response, nil
}

func (m *MockOpenAIClient) AddResponse(response openai.ChatCompletionResponse) {
	m.ChatCompletionResponses = append(m.ChatCompletionResponses, response)
}

func (m *MockOpenAIClient) AddError(err error) {
	m.ChatCompletionErrors = append(m.ChatCompletionErrors, err)
}

func (m *MockOpenAIClient) Reset() {
	m.ChatCompletionResponses = make([]openai.ChatCompletionResponse, 0)
	m.ChatCompletionErrors = make([]error, 0)
	m.CallCount = 0
}

// Helper function to create standard mock responses
func CreateMockResponse(content string, toolCalls []openai.ToolCall) openai.ChatCompletionResponse {
	return openai.ChatCompletionResponse{
		Choices: []openai.ChatCompletionChoice{
			{
				Message: openai.ChatCompletionMessage{
					Role:      openai.ChatMessageRoleAssistant,
					Content:   content,
					ToolCalls: toolCalls,
				},
			},
		},
	}
}

func CreateMockToolCall(id, name, arguments string) openai.ToolCall {
	return openai.ToolCall{
		ID:   id,
		Type: "function",
		Function: openai.FunctionCall{
			Name:      name,
			Arguments: arguments,
		},
	}
}