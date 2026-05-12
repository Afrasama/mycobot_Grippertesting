import pybullet as p
import pybullet_data
import os
import numpy as np

def test_pr2_gripper():
    # Connect with GUI
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
    
    # Load PR2 gripper
    gripper = p.loadURDF("pr2_gripper.urdf")
    
    # Get gripper joint info
    print("PR2 gripper joints:")
    for i in range(p.getNumJoints(gripper)):
        joint_info = p.getJointInfo(gripper, i)
        joint_name = joint_info[1].decode()
        joint_type = joint_info[2]
        print(f"  Joint {i}: {joint_name} (type: {joint_type})")
    
    # Get end effector position and orientation
    ee_state = p.getLinkState(robot, ee_index)
    ee_pos = ee_state[0]
    ee_orn = ee_state[1]
    
    print(f"End effector position: {ee_pos}")
    print(f"End effector orientation: {ee_orn}")
    
    # Position gripper at end effector
    gripper_pos, gripper_orn = p.multiplyTransforms(
        ee_pos, ee_orn,
        [0, 0, 0.05], [0, 0, 0, 1]
    )
    
    p.resetBasePositionAndOrientation(gripper, gripper_pos, gripper_orn)
    
    # Create constraint
    constraint = p.createConstraint(
        robot, ee_index,
        gripper, -1,
        p.JOINT_FIXED,
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    )
    
    p.changeConstraint(constraint, maxForce=100)
    
    # Test gripper control
    print("Testing gripper control...")
    
    # Get finger joints
    finger_joints = []
    for i in range(p.getNumJoints(gripper)):
        joint_info = p.getJointInfo(gripper, i)
        joint_name = joint_info[1].decode()
        if "finger" in joint_name.lower():
            finger_joints.append(i)
    
    print(f"Finger joints: {finger_joints}")
    
    # Open gripper
    for joint_idx in finger_joints:
        p.setJointMotorControl2(
            gripper, joint_idx,
            p.POSITION_CONTROL,
            targetPosition=0.0,
            force=30
        )
    
    # Step simulation
    for _ in range(100):
        p.stepSimulation()
    
    # Close gripper
    for joint_idx in finger_joints:
        p.setJointMotorControl2(
            gripper, joint_idx,
            p.POSITION_CONTROL,
            targetPosition=0.05,
            force=30
        )
    
    # Step simulation
    for _ in range(100):
        p.stepSimulation()
    
    print("Test complete. Close window to exit.")
    
    # Keep simulation running
    try:
        while p.isConnected():
            p.stepSimulation()
    except:
        pass
    
    p.disconnect()

if __name__ == "__main__":
    test_pr2_gripper()
