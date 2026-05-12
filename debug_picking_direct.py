import pybullet as p
import pybullet_data
import os
import time
import numpy as np

def debug_picking_direct():
    # Connect without GUI to see output
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setRealTimeSimulation(0)
    
    p.loadURDF("plane.urdf")
    
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
    
    # Load compact gripper
    gripper_urdf = os.path.join(BASE_DIR, "urdf", "compact_parallel_gripper.urdf")
    gripper = p.loadURDF(gripper_urdf)
    
    # Get gripper joints
    gripper_joints = []
    for i in range(p.getNumJoints(gripper)):
        joint_info = p.getJointInfo(gripper, i)
        joint_name = joint_info[1].decode()
        if "finger_joint" in joint_name:
            gripper_joints.append(i)
    
    print(f"Gripper joints: {gripper_joints}")
    
    # Attach gripper
    ee_state = p.getLinkState(robot, ee_index)
    ee_pos = ee_state[0]
    ee_orn = ee_state[1]
    
    gripper_offset = [0, 0, 0.02]
    gripper_orn_offset = p.getQuaternionFromEuler([0, 0, 0])
    
    gripper_pos, gripper_orn = p.multiplyTransforms(
        ee_pos, ee_orn,
        gripper_offset, gripper_orn_offset
    )
    
    p.resetBasePositionAndOrientation(gripper, gripper_pos, gripper_orn)
    
    constraint = p.createConstraint(
        robot, ee_index,
        gripper, -1,
        p.JOINT_FIXED,
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    )
    
    p.changeConstraint(constraint, maxForce=100)
    print("Gripper attached")
    
    # Create cube
    cube = p.loadURDF("cube_small.urdf", [0.20, 0.10, 0.02])
    
    # Open gripper
    def open_gripper():
        for joint_idx in gripper_joints:
            p.setJointMotorControl2(
                gripper,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=0.0,
                force=20,
                positionGain=0.1
            )
        print("Gripper opened")
    
    # Close gripper
    def close_gripper():
        for joint_idx in gripper_joints:
            p.setJointMotorControl2(
                gripper,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=0.015,
                force=20,
                positionGain=0.1
            )
        print("Gripper closed")
    
    # Move to position
    def move_to_position(target_position, steps=100):
        for step in range(steps):
            joint_angles = p.calculateInverseKinematics(
                robot,
                ee_index,
                target_position,
                maxNumIterations=100,
                residualThreshold=1e-4
            )
            
            for j in range(p.getNumJoints(robot)):
                joint_info = p.getJointInfo(robot, j)
                if joint_info[2] == p.JOINT_REVOLUTE:
                    p.setJointMotorControl2(
                        robot,
                        j,
                        p.POSITION_CONTROL,
                        targetPosition=joint_angles[j],
                        force=400,
                        positionGain=0.08,
                        velocityGain=0.6
                    )
            
            p.stepSimulation()
    
    print("\n=== DEBUG GRASP PROCESS ===")
    
    # Open gripper initially
    open_gripper()
    
    # Move above cube
    pick_position = [0.20, 0.10, 0.15]
    print(f"Moving to position: {pick_position}")
    move_to_position(pick_position, steps=50)
    
    # Move down to cube
    grasp_position = [0.20, 0.10, 0.04]  # Slightly above cube
    print(f"Moving to grasp position: {grasp_position}")
    move_to_position(grasp_position, steps=50)
    
    # Get current positions
    ee_state = p.getLinkState(robot, ee_index)
    ee_pos = ee_state[0]
    cube_pos, _ = p.getBasePositionAndOrientation(cube)
    
    print(f"End effector position: {ee_pos}")
    print(f"Cube position: {cube_pos}")
    print(f"Distance: {np.linalg.norm(np.array(ee_pos) - np.array(cube_pos)):.4f}m")
    
    # Check contacts before closing
    contacts_before = p.getContactPoints(gripper, cube)
    print(f"Contacts before closing: {len(contacts_before)}")
    
    if contacts_before:
        for contact in contacts_before:
            print(f"  Contact on gripper link: {contact[3]}, cube link: {contact[4]}")
    
    # Close gripper
    print("\nClosing gripper...")
    close_gripper()
    
    # Step simulation and check contacts
    for i in range(50):
        p.stepSimulation()
        if i % 10 == 0:
            contacts = p.getContactPoints(gripper, cube)
            print(f"  Step {i}: {len(contacts)} contacts")
    
    # Check contacts after closing
    contacts_after = p.getContactPoints(gripper, cube)
    print(f"Contacts after closing: {len(contacts_after)}")
    
    if contacts_after:
        print("SUCCESS: Gripper is in contact with cube!")
    else:
        print("FAILED: No contact between gripper and cube")
        
        # Additional debugging
        print("\n=== ADDITIONAL DEBUGGING ===")
        
        # Check gripper finger positions
        for joint_idx in gripper_joints:
            joint_state = p.getJointState(gripper, joint_idx)
            joint_pos = joint_state[0]
            print(f"Gripper joint {joint_idx} position: {joint_pos:.4f}")
        
        # Check if gripper is actually at the right position
        ee_state_final = p.getLinkState(robot, ee_index)
        ee_pos_final = ee_state_final[0]
        cube_pos_final, _ = p.getBasePositionAndOrientation(cube)
        
        print(f"Final EE position: {ee_pos_final}")
        print(f"Final cube position: {cube_pos_final}")
        print(f"Final distance: {np.linalg.norm(np.array(ee_pos_final) - np.array(cube_pos_final)):.4f}m")
        
        # Check if cube moved
        cube_movement = np.linalg.norm(np.array(cube_pos_final) - np.array(cube_pos))
        print(f"Cube movement: {cube_movement:.4f}m")
    
    p.disconnect()
    print("Debug complete")

if __name__ == "__main__":
    debug_picking_direct()
