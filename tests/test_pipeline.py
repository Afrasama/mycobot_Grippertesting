import json
import os
import tempfile
import unittest
from unittest.mock import patch

import ik_control
from ik_control import execute_recovery
from llm_planner import choose_recovery_strategy
import utils.logger as logger_module


def _mock_joint_info(joint_type, lower=-1.57, upper=1.57, link_name=b"joint"):
    info = [None] * 13
    info[2] = joint_type
    info[8] = lower
    info[9] = upper
    info[12] = link_name
    return tuple(info)


class PipelineTest(unittest.TestCase):
    def test_failure_logging_and_recovery_execution(self):
        mock_llm_response = {
            "explanation": "Shift the arm and retry the grasp",
            "updates": {
                "x_offset": 0.01,
                "y_offset": -0.01,
            },
            "terminate": False,
            "confidence": 0.91,
        }
        robot_state = {
            "robot_id": 1,
            "ee_index": 2,
            "gripper_id": 99,
            "current_target": [0.30, 0.10, 0.20],
            "joint_angles": [0.0, 0.1, -0.1],
            "gripper_force": 2.5,
            "camera_observation": "data/failures/mock.png",
            "task_stage": "grasp",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = os.path.join(temp_dir, "failure_log.jsonl")
            with patch.object(logger_module, "FAILURE_LOG_PATH", log_path):
                strategy = choose_recovery_strategy(
                    mock_llm_response,
                    failure_type="grasp_failure",
                    robot_state=robot_state,
                )

                self.assertEqual(strategy, "reposition_arm")

                with patch.object(ik_control.p, "getNumJoints", return_value=3), \
                     patch.object(
                         ik_control.p,
                         "getJointInfo",
                         side_effect=[
                             _mock_joint_info(ik_control.p.JOINT_REVOLUTE, link_name=b"joint1"),
                             _mock_joint_info(ik_control.p.JOINT_REVOLUTE, link_name=b"joint2"),
                             _mock_joint_info(ik_control.p.JOINT_REVOLUTE, link_name=b"link6"),
                             _mock_joint_info(ik_control.p.JOINT_REVOLUTE, link_name=b"joint1"),
                             _mock_joint_info(ik_control.p.JOINT_REVOLUTE, link_name=b"joint2"),
                             _mock_joint_info(ik_control.p.JOINT_REVOLUTE, link_name=b"link6"),
                             _mock_joint_info(ik_control.p.JOINT_REVOLUTE, link_name=b"joint1"),
                             _mock_joint_info(ik_control.p.JOINT_REVOLUTE, link_name=b"joint2"),
                             _mock_joint_info(ik_control.p.JOINT_REVOLUTE, link_name=b"link6"),
                             _mock_joint_info(ik_control.p.JOINT_REVOLUTE, link_name=b"joint1"),
                             _mock_joint_info(ik_control.p.JOINT_REVOLUTE, link_name=b"joint2"),
                             _mock_joint_info(ik_control.p.JOINT_REVOLUTE, link_name=b"link6"),
                         ],
                     ), \
                     patch.object(
                         ik_control.p,
                         "calculateInverseKinematics",
                         side_effect=[
                             [0.05, -0.03, 0.02],
                             [0.04, -0.02, 0.01],
                         ],
                     ), \
                     patch.object(ik_control.p, "setJointMotorControl2") as set_joint_mock, \
                     patch.object(ik_control.p, "stepSimulation") as step_mock, \
                     patch("ik_control.time.sleep", return_value=None):
                    execute_recovery(strategy, robot_state)

                self.assertTrue(os.path.exists(log_path))
                with open(log_path, "r", encoding="utf-8") as file_obj:
                    rows = [json.loads(line) for line in file_obj if line.strip()]

                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["failure_type"], "grasp_failure")
                self.assertEqual(rows[0]["strategy_chosen"], "reposition_arm")
                self.assertEqual(rows[0]["robot_state"]["task_stage"], "grasp")
                self.assertEqual(rows[0]["llm_response"]["updates"]["x_offset"], 0.01)
                self.assertGreater(set_joint_mock.call_count, 0)
                self.assertGreater(step_mock.call_count, 0)


if __name__ == "__main__":
    unittest.main()
