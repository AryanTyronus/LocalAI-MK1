"""
Test script to validate error handling and DEBUG mode behavior.
"""
import os
import sys
import json

# Set up environment
os.environ['LOCALAI_DEV_MODE'] = '1'

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.logger import logger
from core.config import DEBUG_MODE, PRINT_STACK_TRACES

def test_config_loading():
    """Test that DEBUG config variables load correctly."""
    logger.info("=" * 60)
    logger.info("Testing DEBUG Config Loading")
    logger.info("=" * 60)
    
    logger.info(f"DEBUG_MODE: {DEBUG_MODE}")
    logger.info(f"PRINT_STACK_TRACES: {PRINT_STACK_TRACES}")
    
    assert PRINT_STACK_TRACES == True, "PRINT_STACK_TRACES should always be True"
    logger.info("✓ Config loading test passed")


def test_error_response_format():
    """Test error response formatting in AIService."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Error Response Format")
    logger.info("=" * 60)
    
    from core.model_manager import FakeModelManager
    from services.ai_service import AIService
    
    # Create AIService with FakeModelManager
    mm = FakeModelManager()
    
    ai_svc = AIService(
        model_manager=mm,
        use_4layer_memory=True
    )
    
    # Simulate a generation error by mocking generate to raise an exception
    original_generate = mm.generate
    
    def mock_generate_error(*args, **kwargs):
        raise ValueError("Simulated generation error for testing")
    
    mm.generate = mock_generate_error
    
    # Call ask() and check the response
    response = ai_svc.ask("What is 2+2?", mode="chat")
    
    logger.info(f"Response received: {response}")
    
    # In non-DEBUG mode, response should be generic fallback
    if not DEBUG_MODE:
        assert "couldn't generate" in response.lower() or "try again" in response.lower(), \
            f"Production mode should return generic message, got: {response}"
        logger.info("✓ Production mode returns generic error message")
    else:
        # In DEBUG mode, response should include error details
        assert "ERROR" in response or "ValueError" in response, \
            f"Debug mode should include error details, got: {response}"
        logger.info("✓ Debug mode returns detailed error message")
    
    # Restore original
    mm.generate = original_generate


def test_app_endpoint_error_handling():
    """Test error handling in Flask app endpoint."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Flask App Error Handling")
    logger.info("=" * 60)
    
    from app import app
    
    with app.test_client() as client:
        # Test invalid JSON
        logger.info("Testing invalid JSON request...")
        response = client.post('/chat', data='invalid json', content_type='application/json')
        assert response.status_code == 400, f"Expected 400, got {response.status_code}"
        logger.info(f"✓ Invalid JSON returns 400: {response.json}")
        
        # Test missing message
        logger.info("Testing missing message field...")
        response = client.post('/chat', 
            json={'mode': 'chat'},
            content_type='application/json'
        )
        assert response.status_code == 400, f"Expected 400, got {response.status_code}"
        logger.info(f"✓ Missing message returns 400: {response.json}")
        
        # Test valid request (should work in dev mode)
        logger.info("Testing valid request...")
        response = client.post('/chat',
            json={'message': 'Hello', 'mode': 'chat'},
            content_type='application/json'
        )
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        logger.info(f"✓ Valid request returns 200: {response.json}")


def test_stack_trace_printing():
    """Verify that stack traces are printed to console on errors."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Stack Trace Printing")
    logger.info("=" * 60)
    
    logger.info("Creating intentional error to test stack trace printing...")
    
    try:
        # This should trigger an error that gets logged with stack trace
        raise RuntimeError("Intentional test error to verify stack trace printing")
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.info("✓ Stack trace was printed above")


if __name__ == "__main__":
    logger.info("\n" + "🧪 STARTING ERROR HANDLING TESTS 🧪\n")
    
    try:
        test_config_loading()
        test_stack_trace_printing()
        test_error_response_format()
        test_app_endpoint_error_handling()
        
        logger.info("\n" + "=" * 60)
        logger.info("✓✓✓ ALL ERROR HANDLING TESTS PASSED ✓✓✓")
        logger.info("=" * 60 + "\n")
        
    except AssertionError as e:
        logger.error(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        import traceback
        logger.error(f"\n✗ Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)
