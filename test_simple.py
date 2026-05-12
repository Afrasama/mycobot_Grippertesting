import pybullet as p
import pybullet_data
import os

def test_simple():
    try:
        # Connect without GUI
        p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        print("Testing WSG50 gripper loading...")
        
        # Check if file exists
        gripper_path = os.path.join(pybullet_data.getDataPath(), "gripper", "wsg50_one_motor_gripper.sdf")
        print(f"Gripper path: {gripper_path}")
        print(f"File exists: {os.path.exists(gripper_path)}")
        
        # Try to load
        gripper = p.loadSDF("gripper/wsg50_one_motor_gripper.sdf")
        print(f"Loaded {len(gripper)} bodies from SDF")
        
        if gripper:
            gripper_id = gripper[0]
            num_joints = p.getNumJoints(gripper_id)
            print(f"Number of joints: {num_joints}")
            
            for i in range(num_joints):
                joint_info = p.getJointInfo(gripper_id, i)
                joint_name = joint_info[1].decode()
                print(f"Joint {i}: {joint_name}")
        
        p.disconnect()
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        try:
            p.disconnect()
        except:
            pass

if __name__ == "__main__":
    test_simple()
