# Error Handling & DEBUG Mode Implementation

## Summary

Enhanced the AIService generation pipeline and Flask app to provide comprehensive error handling with full stack trace printing and optional DEBUG mode for development.

## Changes Made

### 1. Config (config.yaml)
Added a new `debug` section:
```yaml
debug:
  enabled: false  # Set to true to return full error details in API responses
  print_stack_traces: true  # Always print stack traces to console
```

**Note**: `print_stack_traces` is always on (console output). `enabled` toggles whether error details are returned in API responses.

### 2. Core Config (core/config.py)
Exported DEBUG settings:
```python
DEBUG_CONFIG = _config.get('debug', {})
DEBUG_MODE = DEBUG_CONFIG.get('enabled', False)
PRINT_STACK_TRACES = DEBUG_CONFIG.get('print_stack_traces', True)
```

### 3. AIService Error Handling (services/ai_service.py)

#### Model Generation Errors
- **Always print**: Full stack trace to console using `traceback.print_exc()`
- **Always log**: Error details including type, message, duration, and mode
- **In DEBUG mode**: Return error details in response (e.g., `"ERROR: ValueError: Simulated error..."`)
- **In production**: Return generic fallback message

#### Document Retrieval
- Wrapped in try/except to prevent failures from blocking request
- Logs error but continues with empty document context

#### Memory Retrieval
- Wrapped in try/except for both 4-layer and legacy memory paths
- Logs error but continues with empty memory context

#### Agent Mode Tool Execution
- Wrapped with exception handling for tool execution failures
- Logs errors with full stack trace
- Returns detailed error in DEBUG mode, generic error in production

### 4. Flask App Error Handling (app.py)

#### Service Initialization
- Prints full stack trace on initialization failure
- Shows error type, message, and context
- Returns detailed error in DEBUG mode, generic message in production

#### Chat Endpoint
- Catches all exceptions at endpoint level
- Prints full stack trace with context (error type, message, user input, mode)
- Returns detailed JSON error response in DEBUG mode:
  ```json
  {
    "response": "Backend error: ValueError: ...",
    "error": "Error message string",
    "error_type": "ValueError"
  }
  ```
- Returns generic fallback in production mode

### 5. Test Suite (tests/test_error_handling.py)

New comprehensive test file covering:
1. **Config Loading**: Verifies DEBUG config variables load correctly
2. **Stack Trace Printing**: Confirms stack traces print to console
3. **Error Response Format**: 
   - Tests AIService error handling in both production and DEBUG modes
   - Verifies generic message in production, detailed error in DEBUG mode
4. **Flask Endpoint Error Handling**:
   - Invalid JSON requests
   - Missing message field
   - Valid requests with mocked errors

## Behavior Matrix

| Scenario | Stack Trace | Response | Mode |
|----------|-------------|----------|------|
| Model generation fails | ✓ Printed to console | Generic "couldn't generate" | Production |
| Model generation fails | ✓ Printed to console | "ERROR: ValueError: ..." | DEBUG |
| Service init fails | ✓ Printed to console | Generic "initialization failed" | Production |
| Service init fails | ✓ Printed to console | "Service initialization failed: ValueError: ..." | DEBUG |
| Document retrieval fails | ✓ Logged | Continues without docs | Both |
| Memory retrieval fails | ✓ Logged | Continues with empty context | Both |
| Tool execution fails | ✓ Printed | Generic error | Production |
| Tool execution fails | ✓ Printed | "Tool execution failed: ErrorType: ..." | DEBUG |

## Usage

### Development
To enable detailed error responses in API:
1. Open `config.yaml`
2. Set `debug.enabled: true`
3. Errors in responses will include full error details
4. Stack traces always print to console

### Production
Keep `debug.enabled: false` in config:
- Generic fallback messages returned to clients
- Stack traces still printed to server logs (for debugging)
- No internal error details exposed to API consumers

## Testing

Run comprehensive error handling tests:
```bash
LOCALAI_DEV_MODE=1 python tests/test_error_handling.py
```

Expected output:
- ✓ Config loading test passed
- ✓ Stack trace was printed above
- ✓ Production mode returns generic error message
- ✓ Debug mode returns detailed error message
- ✓ Invalid JSON returns 400
- ✓ Missing message returns 400
- ✓ Valid request returns 200
- ✓ ALL ERROR HANDLING TESTS PASSED

## Key Design Decisions

1. **Stack traces always print**: Developers need this for debugging, even in production
2. **Fallback messages don't hide errors**: Errors are logged and can be found in logs
3. **DEBUG flag controls API response content**: Sensitive error details only exposed when explicitly enabled
4. **graceful degradation**: Memory/document failures don't block the request
5. **Comprehensive logging**: All errors logged with context (mode, duration, input) for troubleshooting

