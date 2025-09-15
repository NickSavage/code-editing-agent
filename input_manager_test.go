package main

import (
	"bufio"
	"os"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewInputManager(t *testing.T) {
	im := NewInputManager()
	defer im.Cleanup()

	assert.NotNil(t, im.reader)
	assert.NotNil(t, im.sigChan)
	assert.NotNil(t, im.ctrlCPressed)
	assert.NotNil(t, im.shouldClear)
	assert.NotNil(t, im.shouldExit)
}

func TestInputManager_HandleSignals_SingleCtrlC(t *testing.T) {
	im := NewInputManager()
	defer im.Cleanup()

	// Send single SIGINT
	im.sigChan <- syscall.SIGINT

	// Should receive clear signal
	select {
	case <-im.shouldClear:
		// Expected behavior
	case <-im.shouldExit:
		t.Fatal("Expected clear signal, got exit signal")
	case <-time.After(100 * time.Millisecond):
		t.Fatal("Timeout waiting for clear signal")
	}
}

func TestInputManager_HandleSignals_DoubleCtrlC(t *testing.T) {
	im := NewInputManager()
	defer im.Cleanup()

	// Send first SIGINT
	im.sigChan <- syscall.SIGINT

	// Wait for first signal to be processed
	select {
	case <-im.shouldClear:
		// Expected for first signal
	case <-time.After(100 * time.Millisecond):
		t.Fatal("Timeout waiting for first clear signal")
	}

	// Send second SIGINT quickly (within 2 seconds)
	im.sigChan <- syscall.SIGINT

	// Should receive exit signal
	select {
	case <-im.shouldExit:
		// Expected behavior for double Ctrl-C
	case <-im.shouldClear:
		t.Fatal("Expected exit signal, got clear signal")
	case <-time.After(100 * time.Millisecond):
		t.Fatal("Timeout waiting for exit signal")
	}
}

func TestInputManager_HandleSignals_SlowDoubleCtrlC(t *testing.T) {
	im := NewInputManager()
	defer im.Cleanup()

	// Send first SIGINT
	im.sigChan <- syscall.SIGINT

	// Wait for first signal
	select {
	case <-im.shouldClear:
		// Expected
	case <-time.After(100 * time.Millisecond):
		t.Fatal("Timeout waiting for first signal")
	}

	// Wait longer than 2 seconds, then send second SIGINT
	time.Sleep(50 * time.Millisecond) // Shortened for test speed

	// Manually set lastCtrlC to simulate time passage
	im.lastCtrlC = time.Now().Add(-3 * time.Second)

	im.sigChan <- syscall.SIGINT

	// Should receive clear signal (not exit)
	select {
	case <-im.shouldClear:
		// Expected behavior - treated as new single Ctrl-C
	case <-im.shouldExit:
		t.Fatal("Expected clear signal, got exit signal")
	case <-time.After(100 * time.Millisecond):
		t.Fatal("Timeout waiting for signal")
	}
}

func TestInputManager_GetInput_WithMockReader(t *testing.T) {
	// Create a pipe to simulate stdin
	r, w, err := os.Pipe()
	require.NoError(t, err)
	defer r.Close()
	defer w.Close()

	im := NewInputManager()
	defer im.Cleanup()

	// Replace reader with our pipe
	im.reader = bufio.NewReader(strings.NewReader("test input\n"))

	// Test normal input
	input, ok := im.GetInput()
	assert.True(t, ok)
	assert.Equal(t, "test input", input)
}

func TestInputManager_Cleanup(t *testing.T) {
	im := NewInputManager()

	// Cleanup should not panic
	assert.NotPanics(t, func() {
		im.Cleanup()
	})

	// Multiple cleanups should not panic
	assert.NotPanics(t, func() {
		im.Cleanup()
	})
}