import pybullet as p
import pybullet_data
import os
import numpy as np

def test_compact_gripper():
    # Connect with GUI to see the gripper
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    
    # Load mycobot robot
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    URDF_PATH = os.path.join(BASE_DIR, "urdf", "mycobot_320.urdf")
    
    robot = p.loadURDF(
        URDF_PATH,
        useFixedBase=True,
        flags=p.URDF_USE_INERTIA_FROM_FILE
    )
    
    # Find end effector link
    ee_index = None
    for i in range(p.getNumJoints(robot)):
        if p.getJointInfo(robot, i)[12].decode() == "link6":
            ee_index = i
            break
    
    print(f"End effector index: {ee_index}")
    
    # Load compact parallel gripper
    gripper_urdf = os.path.join(BASE_DIR, "urdf", "compact_parallel_gripper.urdf")
    gripper = p.loadURDF(gripper_urdf)
    
    # Get gripper joint indices
    gripper_joints = []
    for i in range(p.getNumJoints(gripper)):
        joint_info = p.getJointInfo(gripper, i)
        joint_name = joint_info[1].decode()
        print(f"Found gripper joint: {joint_name} at index {i}")
        
        if "finger_joint" in joint_name:
            gripper_joints.append(i)
    
    print(f"Gripper joints: {gripper_joints}")
    
    # Get end effector position and orientation
    ee_state = p.getLinkState(robot, ee_index)
    ee_pos = ee_state[0]
    ee_orn = ee_state[1]
    
    print(f"End effector position: {ee_pos}")
    
    # Position compact gripper at end effector
    gripper_offset = [0, 0, 0.02]
    gripper_orn_offset = p.getQuaternionFromEuler([0, 0, 0])
    
    gripper_pos, gripper_orn = p.multiplyTransforms(
        ee_pos, ee_orn,
        gripper_offset, gripper_orn_offset
    )
    
    print(f"Gripper position: {gripper_pos}")
    
    # Reset gripper position and orientation
    p.resetBasePositionAndOrientation(gripper, gripper_pos, gripper_orn)
    
    # Create fixed constraint to attach gripper to end effector
    constraint = p.createConstraint(
        robot, ee_index,
        gripper, -1,
        p.JOINT_FIXED,
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    )
    
    p.changeConstraint(constraint, maxForce=100)
    
    print("Compact parallel gripper attached to robot end effector")
    
    # Test gripper control
    print("Testing gripper control...")
    
    # Open gripper
    for joint_idx in gripper_joints:
        p.setJointMotorControl2(
            gripper,
            joint_idx,
            p.POSITION_CONTROL,
            targetPosition=0.0,  # Open position
            force=20,
            positionGain=0.1
        )
    
    # Step simulation
    for _ in range(100):
        p.stepSimulation()
    
    print("Gripper opened")
    
    # Close gripper
    for joint_idx in gripper_joints:
        p.setJointMotorControl2(
            gripper,
            joint_idx,
            p.POSITION_CONTROL,
            targetPosition=0.015,  # Close position (max 15mm)
            force=20,
            positionGain=0.1
        )
    
    # Step simulation
    for _ in range(100):
        p.stepSimulation()
    
    print("Gripper closed")
    
    # Add a small cube to test grasping
    cube = p.loadURDF("cube_small.urdf", [0.20, 0.10, 0.02])
    
    print("Test complete. Close window to exit.")
    
    # Keep simulation running
    try:
        while p.isConnected():
            p.stepSimulation()
    except:
        pass
    
    p.disconnect()

if __name__ == "__main__":
    test_compact_gripper()
