import pybullet as p
import pybullet_data
import os

def test_pr2_direct():
    # Connect without GUI
    p.connect(p.DIRECT)
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
    try:
        gripper = p.loadURDF("pr2_gripper.urdf")
        print("PR2 gripper loaded successfully")
    except Exception as e:
        print(f"Error loading PR2 gripper: {e}")
        p.disconnect()
        return
    
    # Get gripper joint info
    print("PR2 gripper joints:")
    finger_joints = []
    for i in range(p.getNumJoints(gripper)):
        joint_info = p.getJointInfo(gripper, i)
        joint_name = joint_info[1].decode()
        joint_type = joint_info[2]
        print(f"  Joint {i}: {joint_name} (type: {joint_type})")
        if "finger" in joint_name.lower():
            finger_joints.append(i)
    
    print(f"Finger joints: {finger_joints}")
    
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
    
    print(f"Gripper position: {gripper_pos}")
    print(f"Gripper orientation: {gripper_orn}")
    
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
    print("Constraint created")
    
    # Test gripper control
    print("Testing gripper control...")
    
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
    
    print("Gripper opened")
    
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
    
    print("Gripper closed")
    
    # Test attachment by moving robot
    print("Testing attachment...")
    
    # Move robot slightly
    p.setJointMotorControl2(
        robot, 1,
        p.POSITION_CONTROL,
        targetPosition=0.5,
        force=100
    )
    
    # Step simulation
    for _ in range(200):
        p.stepSimulation()
    
    # Check positions
    ee_state_after = p.getLinkState(robot, ee_index)
    ee_pos_after = ee_state_after[0]
    gripper_pos_after, _ = p.getBasePositionAndOrientation(gripper)
    
    print(f"EE position after movement: {ee_pos_after}")
    print(f"Gripper position after movement: {gripper_pos_after}")
    
    distance = np.linalg.norm(np.array(ee_pos_after) - np.array(gripper_pos_after))
    print(f"Distance between EE and gripper: {distance}")
    
    if distance < 0.01:
        print("SUCCESS: Gripper is properly attached!")
    else:
        print("FAILED: Gripper attachment failed!")
    
    p.disconnect()
    print("Test completed")

if __name__ == "__main__":
    import numpy as np
    test_pr2_direct()
