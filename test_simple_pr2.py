import pybullet as p
import pybullet_data
import os

def test_simple_pr2():
    # Connect without GUI to see output
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
    gripper = p.loadURDF("pr2_gripper.urdf")
    
    # Get gripper joint indices for PR2
    gripper_joints = []
    gripper_motor_joint = None
    
    for i in range(p.getNumJoints(gripper)):
        joint_info = p.getJointInfo(gripper, i)
        joint_name = joint_info[1].decode()
        print(f"Found joint: {joint_name} at index {i}")
        
        # PR2 gripper joint names
        if "gripper_joint" in joint_name:
            gripper_joints.append(i)
            if "left_gripper" in joint_name or "right_gripper" in joint_name:
                gripper_motor_joint = i
    
    print(f"Gripper joints: {gripper_joints}")
    print(f"Gripper motor joint: {gripper_motor_joint}")
    
    # Get end effector position and orientation
    ee_state = p.getLinkState(robot, ee_index)
    ee_pos = ee_state[0]
    ee_orn = ee_state[1]
    
    print(f"End effector position: {ee_pos}")
    
    # Position gripper at end effector with proper orientation
    gripper_offset = [0, 0, 0.05]
    gripper_orn_offset = p.getQuaternionFromEuler([0, 0, 0])
    
    # Transform gripper position
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
    
    # Set constraint parameters to ensure rigid attachment
    p.changeConstraint(constraint, maxForce=100)
    
    print("PR2 gripper attached to robot end effector")
    
    # Test gripper control
    print("Testing gripper control...")
    
    # Open gripper
    for joint_idx in gripper_joints:
        p.setJointMotorControl2(
            gripper,
            joint_idx,
            p.POSITION_CONTROL,
            targetPosition=0.0,  # Open position
            force=30,
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
            targetPosition=0.05,  # Close position
            force=30,
            positionGain=0.1
        )
    
    # Step simulation
    for _ in range(100):
        p.stepSimulation()
    
    print("Gripper closed")
    
    p.disconnect()
    print("Test completed successfully!")

if __name__ == "__main__":
    test_simple_pr2()
