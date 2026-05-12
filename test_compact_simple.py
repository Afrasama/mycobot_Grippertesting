import pybullet as p
import pybullet_data
import os

def test_compact_simple():
    # Connect without GUI to see output
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    
    # Load compact parallel gripper
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    gripper_urdf = os.path.join(BASE_DIR, "urdf", "compact_parallel_gripper.urdf")
    
    print(f"Loading gripper from: {gripper_urdf}")
    print(f"File exists: {os.path.exists(gripper_urdf)}")
    
    try:
        gripper = p.loadURDF(gripper_urdf)
        print("Compact gripper loaded successfully!")
        
        # Get gripper joint info
        print("Compact gripper joints:")
        for i in range(p.getNumJoints(gripper)):
            joint_info = p.getJointInfo(gripper, i)
            joint_name = joint_info[1].decode()
            joint_type = joint_info[2]
            print(f"  Joint {i}: {joint_name} (type: {joint_type})")
        
        # Test gripper control
        gripper_joints = []
        for i in range(p.getNumJoints(gripper)):
            joint_info = p.getJointInfo(gripper, i)
            joint_name = joint_info[1].decode()
            if "finger_joint" in joint_name:
                gripper_joints.append(i)
        
        print(f"Finger joints: {gripper_joints}")
        
        # Open gripper
        print("Opening gripper...")
        for joint_idx in gripper_joints:
            p.setJointMotorControl2(
                gripper,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=0.0,
                force=20
            )
        
        # Step simulation
        for _ in range(100):
            p.stepSimulation()
        
        # Close gripper
        print("Closing gripper...")
        for joint_idx in gripper_joints:
            p.setJointMotorControl2(
                gripper,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=0.015,
                force=20
            )
        
        # Step simulation
        for _ in range(100):
            p.stepSimulation()
        
        print("Gripper test completed successfully!")
        
    except Exception as e:
        print(f"Error loading or testing compact gripper: {e}")
    
    p.disconnect()

if __name__ == "__main__":
    test_compact_simple()
