import pybullet as p
import pybullet_data
import numpy as np
import time
import os

class RobustPickPlace:
    def __init__(self):
        """Initialize the robust pick and place system with standard parallel gripper"""
        # Connect to PyBullet
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)
        
        # Load environment
        p.loadURDF("plane.urdf")
        
        # Set base directory
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        
        # Load mycobot robot
        URDF_PATH = os.path.join(self.BASE_DIR, "urdf", "mycobot_320.urdf")
        self.robot = p.loadURDF(
            URDF_PATH,
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
        gripper_urdf = os.path.join(self.BASE_DIR, "urdf", "compact_parallel_gripper.urdf")
        self.gripper = p.loadURDF(gripper_urdf)
        
        # Get gripper joint indices for standard parallel gripper
        self.gripper_joints = []
        self.gripper_motor_joint = None
        
        for i in range(p.getNumJoints(self.gripper)):
            joint_info = p.getJointInfo(self.gripper, i)
            joint_name = joint_info[1].decode()
            print(f"Found gripper joint: {joint_name} at index {i}")
            if "finger_joint" in joint_name:
                self.gripper_joints.append(i)
        
        print(f"Standard parallel gripper joints: {self.gripper_joints}")
        
        # Initialize gripper state
        self.gripper_open = True
        self.grasped_object = None
        self.grasp_constraint = None
        
        # Attach gripper to robot
        self._attach_gripper()
        
        # Create objects to manipulate
        self._create_objects()
    
    def _attach_gripper(self):
        """Attach standard parallel gripper to robot end effector"""
        # Get end effector position and orientation
        ee_state = p.getLinkState(self.robot, self.ee_index)
        ee_pos = ee_state[0]
        ee_orn = ee_state[1]
        
        # Position gripper at end effector with proper orientation
        gripper_offset = [0, 0, 0.02]  # 2cm forward offset for standard gripper
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
        
        print("Standard parallel gripper attached to robot end effector")
    
    def _create_objects(self):
        """Create objects to manipulate"""
        # Create cube
        self.cube = p.loadURDF(
            "cube_small.urdf",
            basePosition=[0.30, 0.10, 0.02]
        )
        
        # Create target position indicator
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
    
    def pick_and_place_sequence(self):
        """Execute complete pick and place sequence with improved alignment and stability"""
        # Get actual cube position for better targeting
        cube_pos, _ = p.getBasePositionAndOrientation(self.cube)
        
        # Calculate positions based on actual cube location with direct above approach
        pick_position = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.12]  # 12cm directly above cube
        grasp_position = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.01]  # 1cm above cube (standard approach)
        place_position = [0.10, 0.30, 0.20]  # Target location
        
        print(f"Cube at: {cube_pos}")
        print(f"Direct above pick position: {pick_position}")
        print(f"Standard grasp position: {grasp_position}")
        
        # Step 1: Move directly above object with standard orientation
        print("Moving directly above object...")
        self.open_gripper()
        for _ in range(100):  # More steps for precise positioning
            self.move_to_position(pick_position, slow_mode=False)
            p.stepSimulation()
        
        # Step 2: Slow final approach directly above
        print("Slow final approach directly above...")
        for _ in range(80):  # Slow movement for final approach
            self.move_to_position(grasp_position, slow_mode=True)
            p.stepSimulation()
        
        # Step 3: Attempt grasp with retry mechanism
        print("Attempting grasp with centered alignment...")
        if self.grasp_object(max_retries=3):
            # Step 4: Slow lift with stability
            print("Slow lifting object with stability...")
            lift_position = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.15]
            for _ in range(100):  # Slow lift for stability
                self.move_to_position(lift_position, slow_mode=True)
                p.stepSimulation()
            
            # Step 5: Move to place position
            print("Moving to place position...")
            for _ in range(120):
                self.move_to_position(place_position, slow_mode=False)
                p.stepSimulation()
            
            # Step 6: Slow lower to place
            print("Slow lowering to place...")
            place_down_position = [place_position[0], place_position[1], place_position[2] - 0.05]
            for _ in range(80):  # Slow movement for precise placement
                self.move_to_position(place_down_position, slow_mode=True)
                p.stepSimulation()
            
            # Step 7: Release object
            print("Releasing object...")
            self.release_object()
            
            # Step 8: Slow move up after release
            print("Slow moving up after release...")
            for _ in range(60):  # Slow movement for stability
                self.move_to_position(place_position, slow_mode=True)
                p.stepSimulation()
            
            print("Improved pick and place sequence completed successfully!")
            return True
        else:
            print("Failed to grasp object after all retries")
            return False
    
    def run(self):
        """Main execution loop"""
        print("Starting robust pick and place with standard parallel gripper...")
        
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
