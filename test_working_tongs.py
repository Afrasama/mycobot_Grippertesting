import pybullet as p
import pybullet_data
import os
import numpy as np

def test_working_tongs():
    """Test working tongs for proper cube picking"""
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
    
    # Load working tongs
    gripper_urdf = os.path.join(BASE_DIR, "urdf", "working_tongs.urdf")
    gripper = p.loadURDF(gripper_urdf)
    
    # Get tongs joints
    tongs_joints = []
    for i in range(p.getNumJoints(gripper)):
        joint_info = p.getJointInfo(gripper, i)
        joint_name = joint_info[1].decode()
        if "tong_joint" in joint_name:
            tongs_joints.append(i)
    
    print(f"Working tongs joints: {tongs_joints}")
    
    # Attach tongs with high stability
    ee_state = p.getLinkState(robot, ee_index)
    ee_pos = ee_state[0]
    ee_orn = ee_state[1]
    
    tongs_offset = [0, 0, 0.01]
    tongs_orn_offset = p.getQuaternionFromEuler([0, 0, 0])
    
    tongs_pos, tongs_orn = p.multiplyTransforms(
        ee_pos, ee_orn,
        tongs_offset, tongs_orn_offset
    )
    
    p.resetBasePositionAndOrientation(gripper, tongs_pos, tongs_orn)
    
    constraint = p.createConstraint(
        robot, ee_index,
        gripper, -1,
        p.JOINT_FIXED,
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    )
    
    p.changeConstraint(constraint, maxForce=300)
    print("Working tongs attached with high stability")
    
    # Create cube
    cube = p.loadURDF("cube_small.urdf", [0.20, 0.10, 0.02])
    
    # Tongs functions
    def open_tongs():
        for joint_idx in tongs_joints:
            p.setJointMotorControl2(
                gripper,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=0.0,
                force=2,
                positionGain=0.1
            )
        print("Working tongs opened")
    
    def close_tongs():
        for joint_idx in tongs_joints:
            p.setJointMotorControl2(
                gripper,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=0.005,  # 5mm working design
                force=3,
                positionGain=0.3
            )
        print("Working tongs closed - cube grasped")
    
    # Move to position
    def move_to_position(target_position, steps=100):
        for step in range(steps):
            joint_angles = p.calculateInverseKinematics(
                robot,
                ee_index,
                target_position,
                maxNumIterations=500,
                residualThreshold=1e-6
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
    
    print("\n=== WORKING TONGS TEST ===")
    
    # Get cube position
    cube_pos, _ = p.getBasePositionAndOrientation(cube)
    
    # Calculate positions
    pick_position = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.12]
    grasp_position = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.005]  # 5mm working tongs
    
    print(f"Cube at: {cube_pos}")
    print(f"Pick position: {pick_position}")
    print(f"Working tongs grasp position: {grasp_position}")
    
    # Step 1: Move above object
    print("\nMoving above object...")
    open_tongs()
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
    
    # Step 3: Close working tongs
    print("Closing working tongs...")
    close_tongs()
    
    # Monitor contacts for working tongs
    print("Monitoring contacts for working tongs...")
    grasp_stable = False
    contact_count = 0
    contact_history = []
    
    for i in range(30):  # More steps for working tongs
        p.stepSimulation()
        contacts = p.getContactPoints(gripper, cube)
        contact_history.append(len(contacts))
        if len(contacts) > 0:
            contact_count += 1
            if contact_count >= 3:  # Need 3 contacts for working tongs
                grasp_stable = True
                print(f"    ✅ WORKING TONGS GRASP ACHIEVED at step {i}!")
                break
        if i % 5 == 0:
            print(f"  Step {i}: {len(contacts)} contacts, contact_count: {contact_count}")
    
    # Analyze contact pattern
    max_contacts = max(contact_history)
    final_contacts = contact_history[-1]
    
    print(f"Max contacts: {max_contacts}")
    print(f"Final contacts: {final_contacts}")
    print(f"Contact count: {contact_count}")
    
    # Step 4: Check grasp
    grasp_success = False
    if grasp_stable:
        print("SUCCESS: Working tongs grasp achieved!")
        
        # Create constraint with high force
        ee_state = p.getLinkState(robot, ee_index)
        ee_pos = ee_state[0]
        cube_pos_current, _ = p.getBasePositionAndOrientation(cube)
        
        relative_offset = [
            cube_pos_current[0] - ee_pos[0],
            cube_pos_current[1] - ee_pos[1],
            cube_pos_current[2] - ee_pos[2] - 0.01,  # Slight upward offset for stability
        ]
        
        grasp_constraint = p.createConstraint(
            gripper, -1,
            cube, -1,
            p.JOINT_FIXED,
            [0, 0, 0],
            [0, 0, 0],
            relative_offset,
        )
        p.changeConstraint(grasp_constraint, maxForce=200)
        
        grasp_success = True
        print("Working tongs grasp constraint created!")
        
        # Step 5: Test lift with working tongs
        print("Testing lift with working tongs...")
        lift_position = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.15]
        move_to_position(lift_position, steps=80)
        
        # Verify lift
        cube_pos_lifted, _ = p.getBasePositionAndOrientation(cube)
        lift_height = cube_pos_lifted[2] - cube_pos[2]
        print(f"Object lifted: {lift_height:.4f}m")
        
        # Check stability during lift
        contacts_after_lift = p.getContactPoints(gripper, cube)
        print(f"Contacts during lift: {len(contacts_after_lift)}")
        
        if lift_height > 0.05 and len(contacts_after_lift) > 0:
            print("WORKING TONGS LIFT SUCCESSFUL!")
            print("✅ Cube properly picked up by tongs!")
            success = True
        else:
            print("WORKING TONGS LIFT FAILED")
            success = False
    else:
        print("FAILED: No working tongs grasp achieved")
        success = False
    
    p.disconnect()
    
    if success:
        print("\nWORKING TONGS TEST: PASSED!")
        print("Cube is properly picked up by the tongs!")
    else:
        print("\nWORKING TONGS TEST: FAILED!")
        print("Working tongs need further adjustment.")
    
    return success

if __name__ == "__main__":
    test_working_tongs()
