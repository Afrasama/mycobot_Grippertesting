import pybullet as p
import pybullet_data
import os
import numpy as np

def debug_attachment():
    # Connect with GUI to see what's happening
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
    
    # Load WSG50 gripper
    gripper = p.loadSDF("gripper/wsg50_one_motor_gripper.sdf")[0]
    
    # Get initial positions
    ee_state = p.getLinkState(robot, ee_index)
    ee_pos = ee_state[0]
    ee_orn = ee_state[1]
    
    print(f"Initial EE position: {ee_pos}")
    
    # Try different attachment methods
    print("Testing attachment method 1: Direct constraint...")
    
    # Method 1: Direct constraint without repositioning
    constraint1 = p.createConstraint(
        robot, ee_index,
        gripper, -1,
        p.JOINT_FIXED,
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    )
    
    p.changeConstraint(constraint1, maxForce=100)
    
    # Step simulation and check
    for _ in range(100):
        p.stepSimulation()
    
    gripper_pos1, _ = p.getBasePositionAndOrientation(gripper)
    print(f"Gripper position after method 1: {gripper_pos1}")
    
    # Remove constraint and try method 2
    p.removeConstraint(constraint1)
    
    print("Testing attachment method 2: Reposition then constraint...")
    
    # Method 2: Reposition gripper then create constraint
    p.resetBasePositionAndOrientation(gripper, ee_pos, ee_orn)
    
    constraint2 = p.createConstraint(
        robot, ee_index,
        gripper, -1,
        p.JOINT_FIXED,
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    )
    
    p.changeConstraint(constraint2, maxForce=100)
    
    # Step simulation
    for _ in range(100):
        p.stepSimulation()
    
    gripper_pos2, _ = p.getBasePositionAndOrientation(gripper)
    print(f"Gripper position after method 2: {gripper_pos2}")
    
    # Test movement
    print("Testing movement with method 2...")
    
    # Move robot slightly
    p.setJointMotorControl2(
        robot, 1,
        p.POSITION_CONTROL,
        targetPosition=0.5,
        force=100
    )
    
    for _ in range(200):
        p.stepSimulation()
    
    ee_state_after = p.getLinkState(robot, ee_index)
    ee_pos_after = ee_state_after[0]
    gripper_pos_after, _ = p.getBasePositionAndOrientation(gripper)
    
    print(f"EE position after movement: {ee_pos_after}")
    print(f"Gripper position after movement: {gripper_pos_after}")
    
    distance = np.linalg.norm(np.array(ee_pos_after) - np.array(gripper_pos_after))
    print(f"Distance: {distance}")
    
    if distance < 0.01:
        print("SUCCESS: Gripper follows EE movement!")
    else:
        print("FAILED: Gripper doesn't follow EE movement")
    
    # Keep simulation running for inspection
    print("Simulation running... Close window to exit")
    try:
        while p.isConnected():
            p.stepSimulation()
    except:
        pass
    
    p.disconnect()

if __name__ == "__main__":
    debug_attachment()
