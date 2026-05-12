#!/usr/bin/env python3
"""
Test Ollama integration with the robotics system
"""
import os
import sys
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from reflection.llm_reflection_agent import LLMReflectionAgent

def test_ollama_integration():
    """Test Ollama integration with the LLM Reflection Agent"""
    
    print("Testing Ollama Integration")
    print("=" * 40)
    
    # Configure environment for Ollama
    os.environ["LLM_AGENT_BACKEND"] = "ollama"
    os.environ["LLM_AGENT_ENDPOINT"] = "http://localhost:11434/api/chat"
    os.environ["LLM_AGENT_MODEL"] = "llama3.2-vision"
    os.environ["LLM_AGENT_TIMEOUT_S"] = "120"
    os.environ["LLM_AGENT_USE_VISION"] = "1"
    
    try:
        # Create the LLM agent
        agent = LLMReflectionAgent()
        
        print(f"Backend: {agent.backend}")
        print(f"Model: {agent.model}")
        print(f"Endpoint: {agent.endpoint}")
        print(f"Timeout: {agent.timeout_s}s")
        print(f"Vision: {agent.use_vision}")
        print(f"Configured: {agent.is_configured()}")
        
        if not agent.is_configured():
            print("ERROR: Agent not properly configured")
            return False
            
        # Test with sample robotics failure scenario
        scene_info = {
            "failure_type": "grasp_failure",
            "pixel_error_x": 15.0,
            "pixel_error_y": -8.0,
            "retry_count": 2,
            "object_detected": True,
            "gripper_open": True
        }
        
        current_policy = {
            "x_offset": 0.0,
            "y_offset": 0.0,
            "grasp_height": 0.05,
            "approach_height": 0.15,
            "lift_height": 0.20,
            "release_delay": 60
        }
        
        # Create a dummy RGB image (128x128x3)
        dummy_rgb = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        
        print("\nTesting reflection with sample failure scenario...")
        decision = agent.reflect(scene_info, current_policy, rgb=dummy_rgb)
        
        print(f"\nDecision received:")
        print(f"Explanation: {decision.explanation}")
        print(f"Updates: {decision.updates}")
        print(f"Terminate: {decision.terminate}")
        print(f"Confidence: {decision.confidence}")
        print(f"Mode: {decision.mode}")
        
        print("\n[SUCCESS] Ollama integration test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n[FAILED] Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ollama_integration()
    sys.exit(0 if success else 1)
