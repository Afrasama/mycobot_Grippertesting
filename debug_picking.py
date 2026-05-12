import pybullet as p
import pybullet_data
import os
import time
import numpy as np

class DebugPicking:
    def __init__(self):
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)
        
        p.loadURDF("plane.urdf")
        
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.URDF_PATH = os.path.join(self.BASE_DIR, "urdf", "mycobot_320.urdf")
        
        self.robot = p.loadURDF(
            self.URDF_PATH,
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE
        )
        
        # Find end effector link
        self.ee_index = None
        for i in range(p.getNumJoints(self.robot)):
            if p.getJointInfo(self.robot, i)[12].decode() == "link6":
                self.ee_index = i
                break
        
        print(f"End effector index: {self.ee_index}")
        
        # Load compact gripper
        gripper_urdf = os.path.join(self.BASE_DIR, "urdf", "compact_parallel_gripper.urdf")
        self.gripper = p.loadURDF(gripper_urdf)
        
        # Get gripper joints
        self.gripper_joints = []
        for i in range(p.getNumJoints(self.gripper)):
            joint_info = p.getJointInfo(self.gripper, i)
            joint_name = joint_info[1].decode()
            if "finger_joint" in joint_name:
                self.gripper_joints.append(i)
        
        print(f"Gripper joints: {self.gripper_joints}")
        
        # Attach gripper
        self._attach_gripper()
        
        # Create cube
        self.cube = p.loadURDF("cube_small.urdf", [0.20, 0.10, 0.02])
        
        # Gripper state
        self.gripper_open = True
        self.grasped_object = None
    
    def _attach_gripper(self):
        """Attach compact parallel gripper to robot end effector"""
        ee_state = p.getLinkState(self.robot, self.ee_index)
        ee_pos = ee_state[0]
        ee_orn = ee_state[1]
        
        gripper_offset = [0, 0, 0.02]
        gripper_orn_offset = p.getQuaternionFromEuler([0, 0, 0])
        
        gripper_pos, gripper_orn = p.multiplyTransforms(
            ee_pos, ee_orn,
            gripper_offset, gripper_orn_offset
        )
        
        p.resetBasePositionAndOrientation(self.gripper, gripper_pos, gripper_orn)
        
        constraint = p.createConstraint(
            self.robot, self.ee_index,
            self.gripper, -1,
            p.JOINT_FIXED,
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        )
        
        p.changeConstraint(constraint, maxForce=100)
        print("Gripper attached")
    
    def open_gripper(self):
        """Open gripper"""
        for joint_idx in self.gripper_joints:
            p.setJointMotorControl2(
                self.gripper,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=0.0,
                force=20,
                positionGain=0.1
            )
        self.gripper_open = True
        print("Gripper opened")
    
    def close_gripper(self):
        """Close gripper"""
        for joint_idx in self.gripper_joints:
            p.setJointMotorControl2(
                self.gripper,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=0.015,
                force=20,
                positionGain=0.1
            )
        self.gripper_open = False
        print("Gripper closed")
    
    def move_to_position(self, target_position, steps=100):
        """Move robot to target position"""
        for step in range(steps):
            joint_angles = p.calculateInverseKinematics(
                self.robot,
                self.ee_index,
                target_position,
                maxNumIterations=100,
                residualThreshold=1e-4
            )
            
            for j in range(p.getNumJoints(self.robot)):
                joint_info = p.getJointInfo(self.robot, j)
                if joint_info[2] == p.JOINT_REVOLUTE:
                    p.setJointMotorControl2(
                        self.robot,
                        j,
                        p.POSITION_CONTROL,
                        targetPosition=joint_angles[j],
                        force=400,
                        positionGain=0.08,
                        velocityGain=0.6
                    )
            
            p.stepSimulation()
            time.sleep(1/240)
    
    def debug_grasp(self):
        """Debug grasp process with detailed information"""
        print("\n=== DEBUG GRASP PROCESS ===")
        
        # Get current positions
        ee_state = p.getLinkState(self.robot, self.ee_index)
        ee_pos = ee_state[0]
        cube_pos, _ = p.getBasePositionAndOrientation(self.cube)
        
        print(f"End effector position: {ee_pos}")
        print(f"Cube position: {cube_pos}")
        print(f"Distance: {np.linalg.norm(np.array(ee_pos) - np.array(cube_pos)):.4f}m")
        
        # Check contacts before closing
        contacts_before = p.getContactPoints(self.gripper, self.cube)
        print(f"Contacts before closing: {len(contacts_before)}")
        
        if contacts_before:
            for contact in contacts_before:
                print(f"  Contact on gripper link: {contact[3]}, cube link: {contact[4]}")
        
        # Close gripper
        print("\nClosing gripper...")
        self.close_gripper()
        
        # Step simulation and check contacts
        for i in range(50):
            p.stepSimulation()
            if i % 10 == 0:
                contacts = p.getContactPoints(self.gripper, self.cube)
                print(f"  Step {i}: {len(contacts)} contacts")
        
        # Check contacts after closing
        contacts_after = p.getContactPoints(self.gripper, self.cube)
        print(f"Contacts after closing: {len(contacts_after)}")
        
        if contacts_after:
            print("SUCCESS: Gripper is in contact with cube!")
            return True
        else:
            print("FAILED: No contact between gripper and cube")
            return False
    
    def run_debug(self):
        """Run debug sequence"""
        print("Starting debug picking test...")
        
        # Open gripper initially
        self.open_gripper()
        
        # Move above cube
        pick_position = [0.20, 0.10, 0.15]
        print(f"Moving to position: {pick_position}")
        self.move_to_position(pick_position, steps=50)
        
        # Move down to cube
        grasp_position = [0.20, 0.10, 0.04]  # Slightly above cube
        print(f"Moving to grasp position: {grasp_position}")
        self.move_to_position(grasp_position, steps=50)
        
        # Debug grasp
        success = self.debug_grasp()
        
        if success:
            print("\n=== GRASP SUCCESSFUL ===")
        else:
            print("\n=== GRASP FAILED ===")
        
        print("Debug complete. Close window to exit.")
        
        try:
            while p.isConnected():
                p.stepSimulation()
        except:
            pass
        
        p.disconnect()

if __name__ == "__main__":
    debug = DebugPicking()
    debug.run_debug()
