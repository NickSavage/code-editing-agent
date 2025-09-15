package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	openai "github.com/sashabaranov/go-openai"
)

const MODEL = "anthropic/claude-sonnet-4"

// Tool input structures
type ReadFileInput struct {
	Path string `json:"path" jsonschema_description:"The relative path of a file in the working directory."`
}

type ListDirInput struct {
	Path      string `json:"path" jsonschema_description:"The relative path of a directory in the working directory."`
	Recursive bool   `json:"recursive" jsonschema_description:"Whether to list the directory recursively"`
}

type WriteFileInput struct {
	Path    string `json:"path" jsonschema_description:"The relative path of a file in the working directory."`
	Content string `json:"content" jsonschema_description:"The content to write to the file. This will overwrite the file if it exists."`
}

// Tool handler function type
type ToolHandler func(toolCall openai.ToolCall) openai.ChatCompletionMessage

// InputManager handles user input with signal management
type InputManager struct {
	reader          *bufio.Reader
	sigChan         chan os.Signal
	lastCtrlC       time.Time
	ctrlCPressed    chan bool
	shouldClear     chan bool
	shouldExit      chan bool
}

// NewInputManager creates a new input manager with signal handling
func NewInputManager() *InputManager {
	im := &InputManager{
		reader:      bufio.NewReader(os.Stdin),
		sigChan:     make(chan os.Signal, 1),
		ctrlCPressed: make(chan bool, 1),
		shouldClear: make(chan bool, 1),
		shouldExit:  make(chan bool, 1),
	}
	
	// Set up signal handling for Ctrl-C
	signal.Notify(im.sigChan, syscall.SIGINT)
	
	// Start signal handler goroutine
	go im.handleSignals()
	
	return im
}

// handleSignals processes Ctrl-C signals
func (im *InputManager) handleSignals() {
	for {
		<-im.sigChan
		now := time.Now()
		
		// Check if this is a double Ctrl-C (within 2 seconds)
		if now.Sub(im.lastCtrlC) < 2*time.Second {
			// Double Ctrl-C, exit
			im.shouldExit <- true
			return
		}
		
		// Single Ctrl-C, clear input
		im.lastCtrlC = now
		im.shouldClear <- true
	}
}

// GetInput reads user input with Ctrl-C handling
func (im *InputManager) GetInput() (string, bool) {
	inputChan := make(chan string, 1)
	errorChan := make(chan error, 1)
	
	// Read input in a goroutine
	go func() {
		input, err := im.reader.ReadString('\n')
		if err != nil {
			errorChan <- err
			return
		}
		inputChan <- strings.TrimSuffix(input, "\n")
	}()
	
	// Wait for input, clear signal, or exit signal
	select {
	case input := <-inputChan:
		return input, true
	case <-errorChan:
		return "", false
	case <-im.shouldClear:
		// Clear the current line and return empty string to retry
		fmt.Print("\r\u001b[K") // Clear line
		fmt.Print("\u001b[94mYou\u001b[0m: ")
		return im.GetInput() // Recursive call for new input
	case <-im.shouldExit:
		return "", false
	}
}

// Cleanup stops signal handling
func (im *InputManager) Cleanup() {
	signal.Stop(im.sigChan)
	close(im.sigChan)
}

// Agent represents an AI assistant with tool capabilities
type Agent struct {
	client         *openai.Client
	inputManager   *InputManager
	tools          []openai.Tool
	toolHandlers   map[string]ToolHandler
}

// NewAgent creates a new agent instance
func NewAgent(client *openai.Client, inputManager *InputManager) *Agent {
	agent := &Agent{
		client:       client,
		inputManager: inputManager,
		toolHandlers: make(map[string]ToolHandler),
	}
	
	agent.setupTools()
	return agent
}

// setupTools initializes all tools and their handlers
func (a *Agent) setupTools() {
	a.tools = []openai.Tool{
		a.createReadFileTool(),
		a.createListDirTool(),
		a.createWriteFileTool(),
	}

	a.toolHandlers = map[string]ToolHandler{
		"read_file":     a.handleReadFile,
		"list_dir":      a.handleListDir,
		"write_to_file": a.handleWriteFile,
	}
}

// Tool creation methods
func (a *Agent) createReadFileTool() openai.Tool {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"path": map[string]any{
				"type":        "string",
				"description": "The relative path of a file in the working directory.",
			},
		},
		"required": []string{"path"},
	}

	return openai.Tool{
		Type: openai.ToolTypeFunction,
		Function: &openai.FunctionDefinition{
			Name:        "read_file",
			Description: "Read the contents of a given relative file path. Use this when you want to see what's inside a file. Do not use this with directory names.",
			Parameters:  schema,
		},
	}
}

func (a *Agent) createListDirTool() openai.Tool {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"path": map[string]any{
				"type":        "string",
				"description": "The relative path of a directory in the working directory.",
			},
			"recursive": map[string]any{
				"type":        "boolean",
				"description": "Whether to list the directory recursively",
			},
		},
		"required": []string{"path"},
	}

	return openai.Tool{
		Type: openai.ToolTypeFunction,
		Function: &openai.FunctionDefinition{
			Name:        "list_dir",
			Description: "List the contents of a given relative directory path.",
			Parameters:  schema,
		},
	}
}

func (a *Agent) createWriteFileTool() openai.Tool {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"path": map[string]any{
				"type":        "string",
				"description": "The relative path of a file in the working directory.",
			},
			"content": map[string]any{
				"type":        "string",
				"description": "The content to write to the file. This will overwrite the file if it exists.",
			},
		},
		"required": []string{"path", "content"},
	}

	return openai.Tool{
		Type: openai.ToolTypeFunction,
		Function: &openai.FunctionDefinition{
			Name:        "write_to_file",
			Description: "Write content to a file, overwriting it if it exists.",
			Parameters:  schema,
		},
	}
}

// Tool handler methods
func (a *Agent) handleReadFile(toolCall openai.ToolCall) openai.ChatCompletionMessage {
	var input ReadFileInput
	if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &input); err != nil {
		return a.createErrorResponse(toolCall.ID, fmt.Sprintf("Invalid arguments: %v", err))
	}

	content, err := os.ReadFile(input.Path)
	if err != nil {
		return a.createErrorResponse(toolCall.ID, fmt.Sprintf("Error reading file: %v", err))
	}

	return openai.ChatCompletionMessage{
		Role:       openai.ChatMessageRoleTool,
		Content:    string(content),
		ToolCallID: toolCall.ID,
	}
}

func (a *Agent) handleListDir(toolCall openai.ToolCall) openai.ChatCompletionMessage {
	var input ListDirInput
	if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &input); err != nil {
		return a.createErrorResponse(toolCall.ID, fmt.Sprintf("Invalid arguments: %v", err))
	}

	if input.Recursive {
		return a.handleRecursiveListDir(toolCall.ID, input.Path)
	}

	files, err := os.ReadDir(input.Path)
	if err != nil {
		return a.createErrorResponse(toolCall.ID, fmt.Sprintf("Error reading directory: %v", err))
	}

	var fileList strings.Builder
	for _, file := range files {
		fileList.WriteString(file.Name() + "\n")
	}

	return openai.ChatCompletionMessage{
		Role:       openai.ChatMessageRoleTool,
		Content:    fileList.String(),
		ToolCallID: toolCall.ID,
	}
}

func (a *Agent) handleRecursiveListDir(toolCallID, dirPath string) openai.ChatCompletionMessage {
	var fileList strings.Builder
	
	err := filepath.Walk(dirPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		relPath, _ := filepath.Rel(dirPath, path)
		if relPath != "." {
			fileList.WriteString(relPath + "\n")
		}
		return nil
	})

	if err != nil {
		return a.createErrorResponse(toolCallID, fmt.Sprintf("Error walking directory: %v", err))
	}

	return openai.ChatCompletionMessage{
		Role:       openai.ChatMessageRoleTool,
		Content:    fileList.String(),
		ToolCallID: toolCallID,
	}
}

func (a *Agent) handleWriteFile(toolCall openai.ToolCall) openai.ChatCompletionMessage {
	var input WriteFileInput
	if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &input); err != nil {
		return a.createErrorResponse(toolCall.ID, fmt.Sprintf("Invalid arguments: %v", err))
	}

	// Ensure directory exists
	dir := filepath.Dir(input.Path)
	if dir != "." {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return a.createErrorResponse(toolCall.ID, fmt.Sprintf("Error creating directory: %v", err))
		}
	}

	err := os.WriteFile(input.Path, []byte(input.Content), 0644)
	if err != nil {
		return a.createErrorResponse(toolCall.ID, fmt.Sprintf("Error writing file: %v", err))
	}

	return openai.ChatCompletionMessage{
		Role:       openai.ChatMessageRoleTool,
		Content:    "File written successfully.",
		ToolCallID: toolCall.ID,
	}
}

func (a *Agent) createErrorResponse(toolCallID, errorMsg string) openai.ChatCompletionMessage {
	return openai.ChatCompletionMessage{
		Role:       openai.ChatMessageRoleTool,
		Content:    errorMsg,
		ToolCallID: toolCallID,
	}
}

// processToolCalls handles all tool calls from the assistant
func (a *Agent) processToolCalls(toolCalls []openai.ToolCall) []openai.ChatCompletionMessage {
	var responses []openai.ChatCompletionMessage
	
	for _, toolCall := range toolCalls {
		if toolCall.Type == "function" {
			fmt.Printf("Tool call: %v\n", toolCall.Function.Name)
			
			if handler, exists := a.toolHandlers[toolCall.Function.Name]; exists {
				response := handler(toolCall)
				responses = append(responses, response)
			} else {
				response := a.createErrorResponse(toolCall.ID, fmt.Sprintf("Unknown tool: %v", toolCall.Function.Name))
				responses = append(responses, response)
			}
		}
	}
	
	return responses
}

// createChatCompletion makes a request to the AI model
func (a *Agent) createChatCompletion(ctx context.Context, messages []openai.ChatCompletionMessage) (openai.ChatCompletionMessage, error) {
	fmt.Printf("- Request\n")
	
	resp, err := a.client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model:    MODEL,
		Messages: messages,
		Tools:    a.tools,
	})
	if err != nil {
		return openai.ChatCompletionMessage{}, err
	}
	
	return resp.Choices[0].Message, nil
}

// Run starts the main conversation loop
func (a *Agent) Run(ctx context.Context) error {
	var messages []openai.ChatCompletionMessage

	fmt.Printf("Chat with %v (single ctrl-c to clear input, double ctrl-c to quit)\n", MODEL)

	defer a.inputManager.Cleanup()

	for {
		// Get user input
		fmt.Print("\u001b[94mYou\u001b[0m: ")
		userInput, ok := a.inputManager.GetInput()
		if !ok {
			fmt.Println() // Add newline before exiting
			break
		}

		// Skip empty inputs
		if strings.TrimSpace(userInput) == "" {
			continue
		}

		// Add user message to conversation
		messages = append(messages, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: userInput,
		})

		// Main conversation loop
		for {
			// Get AI response
			assistantMsg, err := a.createChatCompletion(ctx, messages)
			if err != nil {
				return fmt.Errorf("error creating chat completion: %w", err)
			}

			messages = append(messages, assistantMsg)
			
			if assistantMsg.Content != "" {
				fmt.Printf("Assistant: %v\n", assistantMsg.Content)
			}

			// Process any tool calls
			if len(assistantMsg.ToolCalls) > 0 {
				toolResponses := a.processToolCalls(assistantMsg.ToolCalls)
				messages = append(messages, toolResponses...)
			} else {
				// No tool calls, break inner loop to get next user input
				break
			}
		}
	}

	return nil
}

// setupClient creates and configures the OpenAI client
func setupClient() (*openai.Client, error) {
	apiKey := os.Getenv("LLM_KEY")
	baseURL := os.Getenv("LLM_ENDPOINT")

	if baseURL == "" {
		return nil, fmt.Errorf("LLM_ENDPOINT environment variable is required")
	}

	config := openai.DefaultConfig(apiKey)
	config.BaseURL = baseURL

	return openai.NewClientWithConfig(config), nil
}

func main() {
	// Setup client
	client, err := setupClient()
	if err != nil {
		log.Fatal(err)
	}

	// Create input manager
	inputManager := NewInputManager()

	// Create agent
	agent := NewAgent(client, inputManager)

	// Run agent
	if err := agent.Run(context.Background()); err != nil {
		fmt.Printf("Error: %s\n", err.Error())
	}
}