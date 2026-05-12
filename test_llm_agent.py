import os
import sys
import numpy as np

# Ensure reflection module can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from reflection.llm_reflection_agent import LLMReflectionAgent

def test_agent():
    backend = os.getenv('LLM_AGENT_BACKEND', 'ollama')
    model = os.getenv('LLM_AGENT_MODEL', 'llama3.2:3b')
    print(f"Testing LLMReflectionAgent with backend: {backend} model: {model}...")
    agent = LLMReflectionAgent(
        backend=backend,
        model=model,
        timeout_s=float(os.getenv("LLM_AGENT_TIMEOUT_S", "600"))
    )
    
    scene_info = {
        "failure_type": "grasp_failure",
        "retry_count": 0,
        "cube_visible": True,
        "pixel_error_x": 10.0,
        "pixel_error_y": -5.0,
        "distance_to_goal": 0.15
    }
    
    policy = {
        "x_offset": 0.0,
        "y_offset": 0.0,
        "grasp_height": 0.03
    }
    
    print("Calling reflect()... (this may take a few minutes)")
    decision = agent.reflect(scene_info, policy)
    
    print("\n========== DECISION ==========")
    print("Mode:", decision.mode)
    print("Explanation:", decision.explanation)
    print("Updates:", decision.updates)
    print("==============================\n")

if __name__ == "__main__":
    test_agent()
