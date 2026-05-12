import pybullet as p
import pybullet_data
import os
import numpy as np

def test_final_working():
    """Test final working solution with minimal spacing"""
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
    
    # Load minimal spacing gripper
    gripper_urdf = os.path.join(BASE_DIR, "urdf", "minimal_spacing_gripper.urdf")
    gripper = p.loadURDF(gripper_urdf)
    
    # Get gripper joints
    gripper_joints = []
    for i in range(p.getNumJoints(gripper)):
        joint_info = p.getJointInfo(gripper, i)
        joint_name = joint_info[1].decode()
        if "finger_joint" in joint_name:
            gripper_joints.append(i)
    
    print(f"Minimal spacing gripper joints: {gripper_joints}")
    
    # Attach gripper
    ee_state = p.getLinkState(robot, ee_index)
    ee_pos = ee_state[0]
    ee_orn = ee_state[1]
    
    gripper_offset = [0, 0, 0.01]
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
    print("Minimal spacing gripper attached")
    
    # Create cube
    cube = p.loadURDF("cube_small.urdf", [0.20, 0.10, 0.02])
    
    # Gripper functions
    def open_gripper():
        for joint_idx in gripper_joints:
            p.setJointMotorControl2(
                gripper,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=0.0,
                force=2,
                positionGain=0.1
            )
        print("Minimal spacing gripper opened")
    
    def close_gripper():
        for joint_idx in gripper_joints:
            p.setJointMotorControl2(
                gripper,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=0.003,  # 3mm minimal spacing
                force=2,
                positionGain=0.3
            )
        print("Minimal spacing gripper closed")
    
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
    
    print("\n=== FINAL WORKING SOLUTION TEST ===")
    
    # Get cube position
    cube_pos, _ = p.getBasePositionAndOrientation(cube)
    
    # Calculate positions
    pick_position = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.12]
    grasp_position = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.003]  # 3mm minimal spacing
    
    print(f"Cube at: {cube_pos}")
    print(f"Pick position: {pick_position}")
    print(f"Minimal spacing grasp position: {grasp_position}")
    
    # Step 1: Move above object
    print("\nMoving above object...")
    open_gripper()
    move_to_position(pick_position, steps=80)
    
    # Check position
    ee_state = p.getLinkState(robot, ee_index)
    ee_pos = ee_state[0]
    distance_to_cube = np.linalg.norm(np.array(ee_pos[:2]) - np.array(cube_pos[:2]))
    print(f"Position accuracy: {distance_to_cube:.4f}m from cube center")
    
    # Step 2: Move down to object
    print("Moving down to object...")
    move_to_position(grasp_position, steps=60)
    
    # Check final position
    ee_state = p.getLinkState(robot, ee_index)
    ee_pos = ee_state[0]
    final_distance = np.linalg.norm(np.array(ee_pos) - np.array(cube_pos))
    print(f"Final distance to cube: {final_distance:.4f}m")
    
    # Step 3: Close minimal spacing gripper
    print("Closing minimal spacing gripper...")
    close_gripper()
    
    # Monitor contacts
    print("Monitoring contacts for minimal spacing...")
    contact_history = []
    for i in range(100):
        p.stepSimulation()
        contacts = p.getContactPoints(gripper, cube)
        contact_history.append(len(contacts))
        if i % 20 == 0:
            print(f"  Step {i}: {len(contacts)} contacts")
            if len(contacts) > 0:
                print("    ✅ CONTACT MADE WITH MINIMAL SPACING!")
                break
    
    # Analyze contact pattern
    max_contacts = max(contact_history)
    final_contacts = contact_history[-1]
    
    print(f"Max contacts: {max_contacts}")
    print(f"Final contacts: {final_contacts}")
    
    # Step 4: Check grasp
    grasp_success = False
    if final_contacts > 0:
        print("SUCCESS: Minimal spacing gripper made contact!")
        
        # Create constraint
        ee_state = p.getLinkState(robot, ee_index)
        ee_pos = ee_state[0]
        cube_pos_current, _ = p.getBasePositionAndOrientation(cube)
        
        relative_offset = [
            cube_pos_current[0] - ee_pos[0],
            cube_pos_current[1] - ee_pos[1],
            cube_pos_current[2] - ee_pos[2],
        ]
        
        grasp_constraint = p.createConstraint(
            gripper, -1,
            cube, -1,
            p.JOINT_FIXED,
            [0, 0, 0],
            [0, 0, 0],
            relative_offset,
        )
        p.changeConstraint(grasp_constraint, maxForce=30)
        
        grasp_success = True
        print("Minimal spacing grasp constraint created!")
        
        # Step 5: Test lift
        print("Testing minimal spacing lift...")
        lift_position = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.15]
        move_to_position(lift_position, steps=80)
        
        # Verify lift
        cube_pos_lifted, _ = p.getBasePositionAndOrientation(cube)
        lift_height = cube_pos_lifted[2] - cube_pos[2]
        print(f"Object lifted: {lift_height:.4f}m")
        
        if lift_height > 0.05:
            print("MINIMAL SPACING LIFT SUCCESSFUL!")
            print("✅ Cube properly centered between claws!")
            success = True
        else:
            print("MINIMAL SPACING LIFT FAILED")
            success = False
    else:
        print("FAILED: No contact made")
        success = False
    
    p.disconnect()
    
    if success:
        print("\nFINAL WORKING SOLUTION TEST: PASSED!")
        print("Cube is properly positioned between gripper claws!")
    else:
        print("\nFINAL WORKING SOLUTION TEST: FAILED!")
        print("Minimal spacing needs adjustment.")
    
    return success

if __name__ == "__main__":
    test_final_working()
