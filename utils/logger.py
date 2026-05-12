import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILURE_LOG_PATH = os.path.join(PROJECT_ROOT, "data", "failure_log.jsonl")
EXECUTION_LOG_PATH = os.path.join(PROJECT_ROOT, "logs", "execution.log")

def setup_execution_logger():
    """Setup comprehensive execution logger"""
    os.makedirs(os.path.dirname(EXECUTION_LOG_PATH), exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('robot_execution')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(os.path.dirname(EXECUTION_LOG_PATH), f"execution_{timestamp}.log")
    
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Log session start
    logger.info("=" * 80)
    logger.info("ROBOT EXECUTION SESSION STARTED")
    logger.info("=" * 80)
    
    return logger, log_file


def log_failure(
    failure_type: str,
    robot_state: Dict[str, Any],
    llm_response: Any,
    strategy_chosen: str,
) -> None:
    """
    Append a structured failure/recovery record to the JSONL log.
    """
    os.makedirs(os.path.dirname(FAILURE_LOG_PATH), exist_ok=True)

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "failure_type": failure_type,
        "robot_state": robot_state,
        "llm_response": llm_response,
        "strategy_chosen": strategy_chosen,
    }

    with open(FAILURE_LOG_PATH, "a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(record, ensure_ascii=True) + "\n")

def log_robot_state(logger, state: str, details: str = "", attempt: int = 0, distance: float = None):
    """Log robot state changes"""
    distance_str = f"{distance:.3f}m" if distance is not None else "N/A"
    message = f"STATE: {state} | {details} | Attempt: {attempt} | Distance: {distance_str}"
    logger.info(message)

def log_llm_decision(logger, decision):
    """Log LLM decision details"""
    logger.info("=" * 60)
    logger.info("LLM DECISION")
    logger.info("=" * 60)
    logger.info(f"Mode: {decision.mode}")
    logger.info(f"Confidence: {decision.confidence:.3f}" if decision.confidence else "Confidence: N/A")
    logger.info(f"Explanation: {decision.explanation}")
    logger.info(f"Updates: {decision.updates}")
    logger.info("=" * 60)

def log_policy_update(logger, old_policy, new_policy):
    """Log policy changes"""
    logger.info("POLICY UPDATE")
    logger.info(f"Old: {old_policy}")
    logger.info(f"New: {new_policy}")
    logger.info("=" * 40)

def log_session_summary(logger, total_attempts: int, final_distance: float, success: bool):
    """Log session summary"""
    logger.info("=" * 80)
    logger.info("SESSION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total Attempts: {total_attempts}")
    logger.info(f"Final Distance: {final_distance:.3f}m")
    logger.info(f"Success: {'YES' if success else 'NO'}")
    logger.info("=" * 80)
    logger.info("ROBOT EXECUTION SESSION ENDED")
    logger.info("=" * 80)
