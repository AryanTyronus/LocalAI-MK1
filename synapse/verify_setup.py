#!/usr/bin/env python3
"""SYNAPSE Setup Verification - Test all components."""

import sys
import os


def print_header(text):
    """Print section header."""
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print('=' * 60)


def check_python_version():
    """Check Python version."""
    print_header("1. Python Version Check")
    version = sys.version_info
    print(f"Python: {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    print("✓ Python version OK")
    return True


def check_dependencies():
    """Check required dependencies."""
    print_header("2. Dependencies Check")
    
    required = {
        "openai": "OpenAI SDK",
        "dotenv": "Python-dotenv",
    }
    
    all_ok = True
    for module, name in required.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"❌ {name} - NOT INSTALLED")
            print(f"   Install with: pip install {module}")
            all_ok = False
    
    return all_ok


def check_api_key():
    """Check API key configuration."""
    print_header("3. API Key Configuration")
    
    # Check environment variable
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        masked = api_key[:4] + "*" * (len(api_key) - 8) + api_key[-4:]
        print(f"✓ OPENROUTER_API_KEY found: {masked}")
        return True
    
    # Check .env file
    if os.path.exists(".env"):
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key:
            masked = api_key[:4] + "*" * (len(api_key) - 8) + api_key[-4:]
            print(f"✓ API key loaded from .env: {masked}")
            return True
    
    print("❌ DEEPSEEK_API_KEY not found")
    print("   Set with: export OPENROUTER_API_KEY='your-key'")
    print("   Or create .env file with OPENROUTER_API_KEY=your-key")
    return False


def check_synapse_import():
    """Check SYNAPSE module imports."""
    print_header("4. SYNAPSE Module Imports")
    
    try:
        from synapse import SynapseCore
        print("✓ SynapseCore imported")
    except ImportError as e:
        print(f"❌ Failed to import SynapseCore: {e}")
        return False
    
    try:
        from synapse.llm import OpenRouterClient
        print("✓ OpenRouterClient imported")
    except ImportError as e:
        print(f"❌ Failed to import OpenRouterClient: {e}")
        return False
    
    try:
        from synapse.tools import ToolRegistry
        print("✓ ToolRegistry imported")
    except ImportError as e:
        print(f"❌ Failed to import ToolRegistry: {e}")
        return False
    
    try:
        from synapse.memory import ConversationMemory
        print("✓ ConversationMemory imported")
    except ImportError as e:
        print(f"❌ Failed to import ConversationMemory: {e}")
        return False
    
    return True


def check_api_connection():
    """Test DeepSeek API connection."""
    print_header("5. API Connection Test")
    
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("DEEPSEEK_API_KEY")
    
    if not api_key:
        print("⚠ Skipping: API key not available")
        return None
    
    try:
        from synapse.llm import OpenRouterClient
        
        print("Attempting to connect...")
        client = OpenRouterClient(api_key=api_key)
        
        if client.test_connection():
            print("✓ OpenRouter API connection successful")
            return True
        else:
            print("❌ OpenRouter API connection failed")
            return False
    
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False


def check_tool_registry():
    """Check tool registry functionality."""
    print_header("6. Tool Registry Check")
    
    try:
        from synapse.tools import ToolRegistry
        
        registry = ToolRegistry()
        tools = list(registry.tools.keys())
        
        print(f"✓ Tool registry initialized")
        print(f"✓ Registered tools ({len(tools)}):")
        for tool in tools:
            print(f"  - {tool}")
        
        return True
    
    except Exception as e:
        print(f"❌ Tool registry error: {e}")
        return False


def check_memory():
    """Check memory functionality."""
    print_header("7. Memory System Check")
    
    try:
        from synapse.memory import ConversationMemory
        
        memory = ConversationMemory(max_messages=10)
        memory.add_message("user", "Hello")
        memory.add_message("assistant", "Hi there!")
        
        messages = memory.get_messages()
        if len(messages) == 2:
            print("✓ Memory system working")
            print(f"✓ Messages stored: {len(messages)}")
            return True
        else:
            print(f"❌ Expected 2 messages, got {len(messages)}")
            return False
    
    except Exception as e:
        print(f"❌ Memory error: {e}")
        return False


def check_synapse_core():
    """Check SYNAPSE core initialization."""
    print_header("8. SYNAPSE Core Initialization")
    
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("DEEPSEEK_API_KEY")
    
    if not api_key:
        print("⚠ Skipping: API key not available")
        return None
    
    try:
        from synapse import SynapseCore
        
        print("Initializing SynapseCore...")
        synapse = SynapseCore(
            openrouter_api_key=api_key,
            persist_memory=False,
            debug=False
        )
        
        print("✓ SynapseCore initialized successfully")
        
        # Check components
        print(f"✓ LLM client ready")
        print(f"✓ Tool registry ready ({len(synapse.tools.tools)} tools)")
        print(f"✓ Memory ready ({synapse.memory.max_messages} max messages)")
        
        return True
    
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        return False


def print_summary(results):
    """Print test summary."""
    print_header("Summary")
    
    total = len(results)
    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    skipped = sum(1 for r in results.values() if r is None)
    
    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    
    if failed == 0:
        print("\n✅ All checks passed! SYNAPSE is ready to use.")
        return True
    else:
        print("\n❌ Some checks failed. See above for details.")
        return False


def main():
    """Run all verification checks."""
    print("""
╔════════════════════════════════════════════════════════════╗
║         SYNAPSE Setup Verification                        ║
║      Checking all components and dependencies              ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    results = {
        "Python Version": check_python_version(),
        "Dependencies": check_dependencies(),
        "API Key": check_api_key(),
        "Module Imports": check_synapse_import(),
        "API Connection": check_api_connection(),
        "Tool Registry": check_tool_registry(),
        "Memory System": check_memory(),
        "SYNAPSE Core": check_synapse_core(),
    }
    
    success = print_summary(results)
    
    if success:
        print("""
Next steps:
1. Run: python -m synapse.main
2. Try: "Hello, SYNAPSE!"
3. Try: "Search web for 'openrouter api'"
4. Try: /memory
5. Try: /help
    """)
    else:
        print("""
Troubleshooting:
1. Install dependencies: pip install -r synapse/requirements.txt
2. Set API key: export DEEPSEEK_API_KEY='your-key'
3. Check network connection
4. Verify API key is valid at https://openrouter.ai/keys
5. Enable debug mode: --debug flag
    """)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
