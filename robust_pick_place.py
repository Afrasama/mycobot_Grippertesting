import pybullet as p
import pybullet_data
import time
import os

import numpy as np

class RobustPickPlace:
    def __init__(self):
        # Connect to PyBullet
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0) 
        
        # Load environment
        p.loadURDF("plane.urdf")
        
        # Get base directory
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.URDF_PATH = os.path.join(self.BASE_DIR, "urdf", "mycobot_320.urdf")
        
        # Load mycobot robot
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
        
        # Load standard parallel gripper (existing myCobot 320 compatibility)
        gripper_urdf = os.path.join(self.BASE_DIR, "urdf", "pr2_gripper.urdf")
        self.gripper = p.loadURDF(gripper_urdf)
        
        # Get gripper joint indices for compact gripper
        self.gripper_joints = []
        self.gripper_motor_joint = None
        
        for i in range(p.getNumJoints(self.gripper)):
            joint_info = p.getJointInfo(self.gripper, i)
            joint_name = joint_info[1].decode()
            print(f"Found gripper joint: {joint_name} at index {i}")
            
            # Compact gripper joint names
            if "finger_joint" in joint_name:
                self.gripper_joints.append(i)
                if "left_finger" in joint_name or "right_finger" in joint_name:
                    self.gripper_motor_joint = i
        
        print(f"Gripper joints: {self.gripper_joints}")
        print(f"Gripper motor joint: {self.gripper_motor_joint}")
        
        # Attach gripper to robot end effector
        self._attach_gripper()
        
        # Gripper state
        self.gripper_open = True
        self.grasped_object = None
        
        # Object to manipulate
        self.cube = None
        self._create_objects()
        
    def _attach_gripper(self):
        """Attach simple working gripper to robot end effector"""
        # Get end effector position and orientation
        ee_state = p.getLinkState(self.robot, self.ee_index)
        ee_pos = ee_state[0]
        ee_orn = ee_state[1]
        
        # Position simple gripper at end effector with proper orientation
        # Simple gripper should be oriented downward with minimal offset
        gripper_offset = [0, 0, 0.01]  # Minimal forward offset for simple gripper
        gripper_orn_offset = p.getQuaternionFromEuler([0, 0, 0])  # No additional rotation
        
        # Transform gripper position
        gripper_pos, gripper_orn = p.multiplyTransforms(
            ee_pos, ee_orn,
            gripper_offset, gripper_orn_offset
        )
        
        # Reset gripper position and orientation
        p.resetBasePositionAndOrientation(self.gripper, gripper_pos, gripper_orn)
        
        # Create fixed constraint to attach gripper to end effector
        constraint = p.createConstraint(
            self.robot, self.ee_index,
            self.gripper, -1,
            p.JOINT_FIXED,
            [0, 0, 0],
            [0, 0, 0],  # No offset since we positioned it correctly
            [0, 0, 0]
        )
        
        # Set constraint parameters to ensure rigid attachment
        p.changeConstraint(constraint, maxForce=300)
        
        print("Simple working gripper attached to robot end effector")
    
    def _create_objects(self):
        """Create objects to manipulate"""
        # Create cube
        self.cube = p.loadURDF(
            "cube_small.urdf",
            basePosition=[0.30, 0.10, 0.02]
        )
        
        # Create target position marker
        target_position = [0.10, 0.30, 0.20]
        target_vis = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.02,
            rgbaColor=[1, 0, 0, 1]
        )
        p.createMultiBody(baseVisualShapeIndex=target_vis, basePosition=target_position)
        
        return target_position
    
    def open_gripper(self):
        """Open the standard parallel gripper"""
        if not self.gripper_open and len(self.gripper_joints) > 0:
            # Standard parallel gripper has left and right finger joints
            for joint_idx in self.gripper_joints:
                p.setJointMotorControl2(
                    self.gripper,
                    joint_idx,
                    p.POSITION_CONTROL,
                    targetPosition=0.0,  # Open position
                    force=50,
                    positionGain=0.1
                )
            self.gripper_open = True
            print("Standard parallel gripper opened")
    
    def close_gripper(self):
        """Close the standard parallel gripper with improved alignment"""
        if self.gripper_open and len(self.gripper_joints) > 0:
            # Standard parallel gripper has left and right finger joints
            for joint_idx in self.gripper_joints:
                p.setJointMotorControl2(
                    self.gripper,
                    joint_idx,
                    p.POSITION_CONTROL,
                    targetPosition=0.04,  # Close position (standard PR2 gripper)
                    force=50,  # Standard force for stable grasp
                    positionGain=0.2  # Moderate gain for smooth control
                )
            self.gripper_open = False
            print("Standard parallel gripper closed - cube centered")
    
    def move_to_position(self, target_position, orientation=None, max_iterations=200, slow_mode=False):
        """Move robot to target position using inverse kinematics with optional slow mode"""
        if orientation is None:
            orientation = p.getQuaternionFromEuler([0, np.pi, 0])  # Gripper facing down (direct above)
        
        # Calculate inverse kinematics with higher accuracy
        joint_angles = p.calculateInverseKinematics(
            self.robot,
            self.ee_index,
            target_position,
            orientation,
            maxNumIterations=500,  # More iterations for better accuracy
            residualThreshold=1e-6  # Tighter threshold
        )
        
        # Apply joint controls with slow mode option
        position_gain = 0.04 if slow_mode else 0.08  # Slower movement for final approach
        velocity_gain = 0.3 if slow_mode else 0.6  # Slower velocity for final approach
        
        for j in range(p.getNumJoints(self.robot)):
            joint_info = p.getJointInfo(self.robot, j)
            if joint_info[2] == p.JOINT_REVOLUTE:
                p.setJointMotorControl2(
                    self.robot,
                    j,
                    p.POSITION_CONTROL,
                    targetPosition=joint_angles[j],
                    force=400,
                    positionGain=position_gain,
                    velocityGain=velocity_gain
                )
        
        return joint_angles
    
    def check_grasp(self):
        """Check if gripper is in contact with object"""
        if self.cube is None:
            return False
        
        contacts = p.getContactPoints(self.gripper, self.cube)
        return len(contacts) > 0
    
    def grasp_object(self, max_retries=3):
        """Grasp the object with improved alignment and retry mechanism"""
        if self.grasped_object is not None:
            return True
        
        for retry in range(max_retries):
            if retry > 0:
                print(f"Retry attempt {retry + 1}/{max_retries}")
                # Reopen gripper and reposition slightly
                self.open_gripper()
                p.stepSimulation()
                # Slight reposition for retry
                cube_pos, _ = p.getBasePositionAndOrientation(self.cube)
                reposition_offset = [
                    cube_pos[0] + np.random.uniform(-0.005, 0.005),
                    cube_pos[1] + np.random.uniform(-0.005, 0.005),
                    cube_pos[2]
                ]
                self.move_to_position(reposition_offset, slow_mode=True)
                for _ in range(20):
                    p.stepSimulation()
            
            # Close gripper for centering
            self.close_gripper()
            
            # Check for centered grasp with parallel alignment
            grasp_stable = False
            contact_count = 0
            centered_contacts = 0
            
            for i in range(40):  # More steps for stable grasp
                p.stepSimulation()
                contacts = p.getContactPoints(self.gripper, self.cube)
                if len(contacts) > 0:
                    contact_count += 1
                    # Check if contacts are on parallel faces (not corners)
                    parallel_contacts = 0
                    for contact in contacts:
                        # Check if contact normal is mostly vertical (parallel faces)
                        normal = contact[7]
                        if abs(normal[2]) > 0.7:  # Mostly vertical contact
                            parallel_contacts += 1
                    
                    if parallel_contacts >= 2:  # Need parallel face contacts
                        centered_contacts += 1
                    
                    if centered_contacts >= 5:  # Need 5 centered contact checks
                        grasp_stable = True
                        break
            
            if grasp_stable:
                self.grasped_object = self.cube
                # Create constraint with centered offset
                ee_state = p.getLinkState(self.robot, self.ee_index)
                ee_pos = ee_state[0]
                cube_pos, _ = p.getBasePositionAndOrientation(self.cube)
                
                # Calculate centered offset for constraint
                relative_offset = [
                    cube_pos[0] - ee_pos[0],
                    cube_pos[1] - ee_pos[1],
                    cube_pos[2] - ee_pos[2] - 0.005,  # Slight upward offset for stability
                ]
                
                self.grasp_constraint = p.createConstraint(
                    self.gripper, -1,
                    self.cube, -1,
                    p.JOINT_FIXED,
                    [0, 0, 0],
                    [0, 0, 0],
                    relative_offset,
                )
                p.changeConstraint(self.grasp_constraint, maxForce=100)  # Moderate force for stability
                print(f"Object grasped with centered alignment! (Attempt {retry + 1})")
                return True
            else:
                print(f"Grasp failed on attempt {retry + 1}, contacts: {contact_count}, centered: {centered_contacts}")
        
        print("Failed to grasp object after all retries")
        return False
    
    def release_object(self):
        """Release the grasped object"""
        if self.grasped_object is not None:
            p.removeConstraint(self.grasp_constraint)
            self.grasped_object = None
            print("Object released")
    
    def grasp_object(self, max_retries=3):
        """Grasp the object with improved alignment and retry mechanism"""
        if self.grasped_object is not None:
            return True
        
        for retry in range(max_retries):
            if retry > 0:
                print(f"Retry attempt {retry + 1}/{max_retries}")
                # Reopen gripper and reposition slightly
                self.open_gripper()
                p.stepSimulation()
                # Slight reposition for retry
                cube_pos, _ = p.getBasePositionAndOrientation(self.cube)
                reposition_offset = [
                    cube_pos[0] + np.random.uniform(-0.005, 0.005),
                    cube_pos[1] + np.random.uniform(-0.005, 0.005),
                    cube_pos[2]
                ]
                self.move_to_position(reposition_offset, slow_mode=True)
                for _ in range(20):
                    p.stepSimulation()
        
        # Close gripper for centering
            self.close_gripper()
            
            # Check for centered grasp with parallel alignment
            grasp_stable = False
            contact_count = 0
            centered_contacts = 0
                
            for i in range(40):  # More steps for stable grasp
                p.stepSimulation()
                contacts = p.getContactPoints(self.gripper, self.cube)
                if len(contacts) > 0:
                    contact_count += 1
                    # Check if contacts are on parallel faces (not corners)
                    parallel_contacts = 0
                    for contact in contacts:
                        # Check if contact normal is mostly vertical (parallel faces)
                        normal = contact[7]
                        if abs(normal[2]) > 0.7:  # Mostly vertical contact
                            parallel_contacts += 1
                    
                    if parallel_contacts >= 2:  # Need parallel face contacts
                        centered_contacts += 1
                    
                    if centered_contacts >= 5:  # Need 5 centered contact checks
                        grasp_stable = True
                        break
                
            if grasp_stable:
                self.grasped_object = self.cube
                # Create constraint with centered offset
                ee_state = p.getLinkState(self.robot, self.ee_index)
                ee_pos = ee_state[0]
                cube_pos, _ = p.getBasePositionAndOrientation(self.cube)
                
                # Calculate centered offset for constraint
                relative_offset = [
                    cube_pos[0] - ee_pos[0],
                    cube_pos[1] - ee_pos[1],
                    cube_pos[2] - ee_pos[2] - 0.005,  # Slight upward offset for stability
                ]
                
                self.grasp_constraint = p.createConstraint(
                    self.gripper, -1,
                    self.cube, -1,
                    p.JOINT_FIXED,
                    [0, 0, 0],
                    [0, 0, 0],
                    relative_offset,
            )
            p.changeConstraint(self.grasp_constraint, maxForce=100)  # Moderate force for stability
            print(f"Object grasped with centered alignment! (Attempt {retry + 1})")
            return True
        else:
            print(f"Grasp failed on attempt {retry + 1}, contacts: {contact_count}, centered: {centered_contacts}")
            print("Failed to grasp object")
            # Try to adjust position and retry
            print("Adjusting position and retrying...")
            adjusted_grasp = [cube_pos[0] + 0.01, cube_pos[1] + 0.01, grasp_position[2]]
            for _ in range(40):
                self.move_to_position(adjusted_grasp)
                p.stepSimulation()
                time.sleep(1/240)
            
            self.close_gripper()
            for _ in range(60):
                p.stepSimulation()
                time.sleep(1/240)
            
            if self.grasp_object():
                print("Retry successful!")
            else:
                print("Retry failed")
    
    def run(self):
        """Main execution loop"""
        print("Starting robust pick and place with WSG50 electric parallel gripper...")
        
        # Run pick and place sequence
        self.pick_and_place_sequence()
        
        # Keep simulation running
        try:
            while p.isConnected():
                p.stepSimulation()
                time.sleep(1/240)
        except KeyboardInterrupt:
            print("Simulation stopped")

if __name__ == "__main__":
    robot_controller = RobustPickPlace()
    robot_controller.run()
