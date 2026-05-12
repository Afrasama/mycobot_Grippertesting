import pybullet as p
import pybullet_data
import time

def test_wsg50_gripper():
    # Connect to PyBullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    
    try:
        # Try to load WSG50 gripper
        print("Loading WSG50 gripper...")
        gripper = p.loadSDF("gripper/wsg50_one_motor_gripper.sdf")[0]
        print("WSG50 gripper loaded successfully!")
        
        # Get gripper info
        num_joints = p.getNumJoints(gripper)
        print(f"Number of joints: {num_joints}")
        
        for i in range(num_joints):
            joint_info = p.getJointInfo(gripper, i)
            joint_name = joint_info[1].decode()
            joint_type = joint_info[2]
            print(f"Joint {i}: {joint_name} (type: {joint_type})")
        
        # Test gripper control
        print("\nTesting gripper control...")
        for _ in range(100):
            p.stepSimulation()
            time.sleep(1/240)
        
    except Exception as e:
        print(f"Error loading WSG50: {e}")
        
        # Try alternative gripper
        try:
            print("Trying PR2 gripper...")
            gripper = p.loadURDF("pr2_gripper.urdf")
            print("PR2 gripper loaded successfully!")
        except Exception as e2:
            print(f"Error loading PR2 gripper: {e2}")
    
    # Keep simulation running
    try:
        while p.isConnected():
            p.stepSimulation()
            time.sleep(1/240)
    except KeyboardInterrupt:
        print("Simulation stopped")

if __name__ == "__main__":
    test_wsg50_gripper()
