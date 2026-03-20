#!/usr/bin/env python3
"""SYNAPSE CLI - Command-line interface for the AI assistant."""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Optional
from getpass import getpass

from synapse.core import SynapseCore


def print_banner():
    """Print SYNAPSE banner."""
    banner = """
╔════════════════════════════════════════════════════════════╗
║                     SYNAPSE v1.0                           ║
║          JARVIS-Style AI Assistant with DeepSeek           ║
╚════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_response(result: dict):
    """Pretty print response from SYNAPSE."""
    print("\n" + "─" * 60)

    if result.get("thinking"):
        print(f"\n💭 Thinking:\n{result['thinking']}\n")

    print(f"🤖 Response:\n{result['response']}\n")

    if result.get("action"):
        print(f"🔧 Action Executed: {result['action']}")
        if result.get("action_result"):
            print(f"📊 Result: {result['action_result']}")

    print("─" * 60 + "\n")


def print_help():
    """Print CLI help."""
    help_text = """
Available Commands:
  /help          - Show this help message
  /memory        - Show conversation memory summary
  /clear         - Clear conversation history
  /export        - Export conversation memory
  /status        - Test LLM connection
  /exit          - Exit SYNAPSE

Regular Input:
  Type any query to SYNAPSE. Responses include reasoning and actions.
    """
    print(help_text)


def handle_command(synapse: SynapseCore, command: str):
    """Handle special commands."""
    if command == "/help":
        print_help()
    elif command == "/memory":
        summary = synapse.get_memory_summary()
        print(f"\n📚 Memory Summary:\n{json.dumps(summary, indent=2)}\n")
    elif command == "/clear":
        synapse.clear_memory()
        print("✓ Memory cleared\n")
    elif command == "/export":
        memory_data = {
            "messages": synapse.memory.get_messages(include_thinking=True),
            "summary": synapse.get_memory_summary(),
        }
        with open("synapse_export.json", "w") as f:
            json.dump(memory_data, f, indent=2)
        print("✓ Memory exported to synapse_export.json\n")
    elif command == "/status":
        connected = synapse.test_connection()
        status = "✓ Connected" if connected else "✗ Connection failed"
        print(f"\n{status}\n")
    elif command == "/exit":
        return False
    else:
        print(f"Unknown command: {command}. Type '/help' for available commands.\n")
    return True


def main():
    """Main CLI loop."""
    def main():\n        \"\"\"Main CLI loop.\"\"\"\n        parser = argparse.ArgumentParser(description="SYNAPSE - AI Assistant with Nemotron via OpenRouter")
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenRouter API key (or set OPENROUTER_API_KEY env var)",
    )
    parser.add_argument(
        "--persist",
        action="store_true",
        help="Persist conversation memory to disk",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Custom system prompt file path",
    )

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        api_key = getpass("Enter OpenRouter API key: ")
    if not api_key:
        print("✗ API key required")
        sys.exit(1)

    # Load custom system prompt if provided
    system_prompt = None
    if args.system_prompt:
        try:
            with open(args.system_prompt, "r") as f:
                system_prompt = f.read()
        except Exception as e:
            print(f"✗ Failed to load system prompt: {e}")
            sys.exit(1)

    # Initialize SYNAPSE
    try:
        print_banner()
        print("🚀 Initializing SYNAPSE...")

        synapse = SynapseCore(
            openrouter_api_key=api_key,
            system_prompt=system_prompt,
            memory_max_messages=20,
            persist_memory=args.persist,
            debug=args.debug,
        )

        # Test connection
        print("🔗 Testing API connection...")
        if not synapse.test_connection():
            print("✗ Failed to connect to OpenRouter API")
        sys.exit(1)

        print("✓ Connected successfully!\n")
        print_help()

    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        sys.exit(1)

    # Main CLI loop
    try:
        while True:
            try:
                user_input = input("\n👤 You: ").strip()

                if not user_input:
                    continue

                # Handle special commands
                if user_input.startswith("/"):
                    if not handle_command(synapse, user_input):
                        break
                    else:
                        continue

                # Process regular query
                print("\n⏳ Processing...")
                result = synapse.process_query(user_input)

                if result["complete"]:
                    print_response(result)
                else:
                    print(f"\n✗ Error: {result['response']}\n")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n✗ Error: {e}\n")

    except EOFError:
        print("\n\n👋 Goodbye!")


if __name__ == "__main__":
    main()
