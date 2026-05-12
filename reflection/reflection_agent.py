import numpy as np


class ReflectionAgent:
    def __init__(self, scale=0.00015, max_step=0.02, swap_axes=False):
        """
        scale: pixel → world gain
        max_step: maximum correction per retry
        swap_axes: if True, use (-py, -px) mapping (useful for some camera setups).
        """
        self.scale = scale
        self.max_step = max_step
        self.swap_axes = swap_axes

    def reflect(self, scene_info):
        """
        scene_info should contain:
        {
            "pixel_error_x": float,
            "pixel_error_y": float,
            "retry_count": int
        }
        """

        px = scene_info["pixel_error_x"]
        py = scene_info["pixel_error_y"]

        explanation = self._generate_explanation(px, py)

        adjust_x, adjust_y = self._decide_correction(px, py)

        return {
            "explanation": explanation,
            "action": {
                "adjust_x": adjust_x,
                "adjust_y": adjust_y
            }
        }

    # --------------------------
    # pseudo-VLM reasoning layer
    # --------------------------

    def _generate_explanation(self, px, py):

        explanation_parts = []

        if abs(px) > 15:
            if px > 0:
                explanation_parts.append("gripper is left of the cube")
            else:
                explanation_parts.append("gripper is right of the cube")

        if abs(py) > 15:
            if py > 0:
                explanation_parts.append("gripper is above the cube")
            else:
                explanation_parts.append("gripper is below the cube")

        if not explanation_parts:
            explanation_parts.append("gripper is well aligned but grasp height may be incorrect")

        return " and ".join(explanation_parts)

    # --------------------------
    # action policy update
    # --------------------------

    def _decide_correction(self, px, py):

        # default mapping: pixel error in x/y directly corrects world x/y
        # (if your camera frame is rotated relative to world, enable swap_axes)
        if self.swap_axes:
            correction_x = -py * self.scale
            correction_y = -px * self.scale
        else:
            correction_x = px * self.scale
            correction_y = -py * self.scale

        # clip correction
        correction_x = np.clip(correction_x, -self.max_step, self.max_step)
        correction_y = np.clip(correction_y, -self.max_step, self.max_step)

        return correction_x, correction_y