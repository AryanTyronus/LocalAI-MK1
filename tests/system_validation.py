import sys
import json
import traceback
from pprint import pprint

# Ensure project root is on path
sys.path.insert(0, '.')

from core.model_manager import ModelManager

# Create a fake model manager to avoid heavy downloads
class FakeModelManager:
    def __init__(self):
        self.tokenizer = None
        self._embed_dim = 8

    def format_chat(self, system, user, history=None):
        # Simple concatenation
        return (system or '') + "\n\n" + (user or '')

    def generate(self, prompt, max_tokens=300, temperature=0.7):
        # Return deterministic outputs depending on prompt contents
        if 'OPEN_APP_TEST' in prompt:
            return json.dumps({
                'action': 'open_app',
                'parameters': {'app_name': 'Safari'}
            })
        if 'CODE_TEST' in prompt:
            return 'def hello():\n    return "hello"'
        # default reply
        return 'This is a fake model response.'

    def embed(self, texts):
        # Return fixed-dimension embeddings
        import numpy as np
        return np.array([[0.0]*self._embed_dim for _ in texts], dtype=float)


# Inject fake instance before other imports
ModelManager._instance = FakeModelManager()

# Now import the dependency container and services
from core.dependency_container import DependencyContainer

results = {'structural': [], 'runtime': [], 'memory': [], 'errors': []}

try:
    container = DependencyContainer()
    ai = container.get_ai_service()
    results['structural'].append('DependencyContainer initialized')
except Exception as e:
    results['errors'].append(f'DependencyContainer init failed: {e}')
    traceback.print_exc()
    print(json.dumps(results, indent=2))
    sys.exit(1)

# Structural checks
try:
    from core.mode_controller import ModeController
    modes = ['chat', 'coding', 'research', 'agent']
    registered = []
    for m in modes:
        mode_obj = ModeController.get_mode(m)
        registered.append((m, type(mode_obj).__name__))
    results['structural'].append({'modes_registered': registered})

    # ToolRegistry
    from core.tool_registry import ToolRegistry
    tools_listed = list(ToolRegistry._tools.keys())
    results['structural'].append({'tools_registered': tools_listed})

    # Memory manager
    if hasattr(ai, 'memory_manager'):
        results['structural'].append('memory_manager ok')
    else:
        results['structural'].append('memory_manager missing')

    # Config check
    from core.config import get_mode_config
    cfg_chat = get_mode_config('chat')
    results['structural'].append({'mode_config_chat_present': bool(cfg_chat)})

except Exception as e:
    results['errors'].append(f'Structural checks failed: {e}')
    traceback.print_exc()

# Runtime Tests
try:
    # 1) Chat Mode
    reply = ai.ask('Hello, how are you?', mode='chat')
    results['runtime'].append({'chat_reply': reply})

    # Memory accumulation
    short_before = len(ai.memory_manager.get_short_term_messages()) if hasattr(ai, 'memory_manager') else None
    ai.ask('I like pizza', mode='chat')
    short_after = len(ai.memory_manager.get_short_term_messages()) if hasattr(ai, 'memory_manager') else None
    results['memory'].append({'short_term_before': short_before, 'short_term_after': short_after})

    # 2) Research Mode - inject a fake document chunk
    dm = ai.doc_manager
    dm.chunks.append({'id': 0, 'doc_id': 'test.pdf', 'doc_name': 'test.pdf', 'page_number': 1, 'chunk_index': 0, 'text': 'This is a test document chunk about quantum entanglement.'})
    dm._build_index()
    r_reply = ai.ask('What is quantum entanglement?', mode='research')
    results['runtime'].append({'research_reply': r_reply})

    # 3) Coding Mode - deterministic low temp
    c_reply = ai.ask('Please write a simple function (CODE_TEST)', mode='coding')
    results['runtime'].append({'coding_reply': c_reply})

    # 4) Agent Mode - tool JSON output and execution
    a_reply = ai.ask('Please open the browser (OPEN_APP_TEST)', mode='agent')
    results['runtime'].append({'agent_reply': a_reply})

except Exception as e:
    results['errors'].append(f'Runtime tests failed: {e}')
    traceback.print_exc()

# Memory Validation
try:
    # Short-term exists
    st = ai.memory_manager.get_short_term_messages()
    results['memory'].append({'short_term_count': len(st)})

    # Rolling summary trigger: push many messages to exceed threshold
    threshold = ai.memory_manager.summary_trigger
    for i in range(threshold + 1):
        ai.memory_manager.add_short_term_message('user', f'Auto message {i}')
    created = ai.memory_manager.maybe_create_summary()
    results['memory'].append({'rolling_summary_created': created, 'rolling_summary_count': len(ai.memory_manager.rolling_summaries)})

    # Semantic memory add and search
    ai.memory_manager.add_semantic_memory('Important fact about X')
    sem_results = ai.memory_manager.search_semantic_memory('Important')
    results['memory'].append({'semantic_search_count': len(sem_results)})

    # Structured memory isolation
    structured = ai.memory_manager.get_structured_memory()
    results['memory'].append({'structured_keys': list(structured.keys())})

except Exception as e:
    results['errors'].append(f'Memory validation failed: {e}')
    traceback.print_exc()

# Final report
print(json.dumps(results, indent=2))

