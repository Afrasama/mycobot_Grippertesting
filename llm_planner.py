from typing import Any, Dict

from utils.logger import log_failure


VALID_STRATEGIES = ["retry_grasp", "reposition_arm", "reset_to_home", "abort"]


def choose_recovery_strategy(
    llm_response: Dict[str, Any],
    failure_type: str,
    robot_state: Dict[str, Any],
) -> str:
    """
    Map an LLM reflection response to a discrete recovery strategy.

    Falls back to ``reset_to_home`` for missing or malformed responses.
    The selected strategy is always logged.
    """
    strategy = "reset_to_home"

    try:
        if not isinstance(llm_response, dict):
            raise TypeError("llm_response must be a dict")

        if bool(llm_response.get("terminate", False)):
            strategy = "abort"
        else:
            explanation = str(llm_response.get("explanation", "")).lower()
            updates = llm_response.get("updates", {})
            confidence = llm_response.get("confidence")

            if not isinstance(updates, dict):
                updates = {}

            x_update = _safe_float(updates.get("x_offset", 0.0))
            y_update = _safe_float(updates.get("y_offset", 0.0))
            grasp_update = _safe_float(updates.get("grasp_height", 0.0))
            approach_update = _safe_float(updates.get("approach_height", 0.0))
            lift_update = _safe_float(updates.get("lift_height", 0.0))
            release_update = _safe_float(updates.get("release_delay", 0.0))
            confidence_value = _safe_float(confidence, default=0.0)

            if any(
                abs(value) > 0.0
                for value in (x_update, y_update, approach_update, lift_update)
            ):
                strategy = "reposition_arm"
            elif abs(grasp_update) > 0.0 or abs(release_update) > 0.0:
                strategy = "retry_grasp"
            elif "collision" in explanation or "unreachable" in explanation:
                strategy = "reset_to_home"
            elif "retry" in explanation or "grasp" in explanation:
                strategy = "retry_grasp"
            elif confidence_value < 0.2 and explanation:
                strategy = "reset_to_home"

    except Exception:
        strategy = "reset_to_home"

    log_failure(
        failure_type=failure_type,
        robot_state=robot_state,
        llm_response=llm_response,
        strategy_chosen=strategy,
    )
    return strategy


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
