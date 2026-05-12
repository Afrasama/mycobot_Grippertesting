import pybullet as p
import pybullet_data
import os
import numpy as np

def test_gripper_attachment():
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
    
    # Load WSG50 gripper
    gripper = p.loadSDF("gripper/wsg50_one_motor_gripper.sdf")[0]
    
    # Get end effector position and orientation
    ee_state = p.getLinkState(robot, ee_index)
    ee_pos = ee_state[0]
    ee_orn = ee_state[1]
    
    print(f"End effector position: {ee_pos}")
    print(f"End effector orientation: {ee_orn}")
    
    # Position gripper at end effector
    gripper_offset = [0, 0, 0.05]
    gripper_pos = [ee_pos[i] + gripper_offset[i] for i in range(3)]
    
    print(f"Gripper position: {gripper_pos}")
    
    # Reset gripper position and orientation
    p.resetBasePositionAndOrientation(gripper, gripper_pos, ee_orn)
    
    # Create fixed constraint
    constraint = p.createConstraint(
        robot, ee_index,
        gripper, -1,
        p.JOINT_FIXED,
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    )
    
    p.changeConstraint(constraint, maxForce=50)
    
    # Test by moving robot and checking gripper position
    print("\nTesting attachment...")
    
    # Move robot joints
    for i in range(p.getNumJoints(robot)):
        joint_info = p.getJointInfo(robot, i)
        if joint_info[2] == p.JOINT_REVOLUTE:
            p.setJointMotorControl2(
                robot,
                i,
                p.POSITION_CONTROL,
                targetPosition=0.5,
                force=100
            )
    
    # Step simulation
    for _ in range(100):
        p.stepSimulation()
    
    # Check positions after movement
    ee_state_after = p.getLinkState(robot, ee_index)
    ee_pos_after = ee_state_after[0]
    
    gripper_pos_after, gripper_orn_after = p.getBasePositionAndOrientation(gripper)
    
    print(f"End effector position after movement: {ee_pos_after}")
    print(f"Gripper position after movement: {gripper_pos_after}")
    
    # Calculate distance
    distance = np.linalg.norm(np.array(ee_pos_after) - np.array(gripper_pos_after))
    print(f"Distance between EE and gripper: {distance}")
    
    if distance < 0.01:
        print("✓ Gripper is properly attached!")
    else:
        print("✗ Gripper attachment failed!")
    
    p.disconnect()

if __name__ == "__main__":
    test_gripper_attachment()
