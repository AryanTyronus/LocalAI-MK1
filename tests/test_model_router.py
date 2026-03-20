#!/usr/bin/env python3
""" 
Test for ModelRouter - verifies routing logic.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, '..')

from core.model_router import ModelRouter

def main():
    router = ModelRouter()
    
    # Test 1: analysis task (smart model)
    print("\n=== Test 1: Analysis Task ===")
    messages1 = [{"role": "user", "content": "Analyze this code snippet for bugs."}]
    result1 = router.generate("analysis", messages1)
    print("Analysis result keys:", list(result1.keys()))
    
    # Test 2: fast task (fallback to smart)
    print("\n=== Test 2: Fast Task ===")
    messages2 = [{"role": "user", "content": "What time is it?"}]
    result2 = router.generate("fast", messages2)
    print("Fast result keys:", list(result2.keys()))
    
    print("\n✅ ModelRouter tests complete - check console for routing logs.")

if __name__ == "__main__":
    main()
