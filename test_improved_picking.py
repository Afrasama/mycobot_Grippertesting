import pybullet as p
import pybullet_data
import os
import time
import numpy as np

class TestImprovedPicking:
    def __init__(self):
        p.connect(p.DIRECT)
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
        self.grasp_constraint = None
    
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
                targetPosition=0.020,  # Updated to 20mm
                force=30,  # Increased force
                positionGain=0.2  # Higher gain
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
    
    def grasp_object(self):
        """Grasp the object with improved detection"""
        if self.grasped_object is not None:
            return True
        
        # Check for contacts multiple times to ensure stable grasp
        grasp_stable = False
        for _ in range(10):
            p.stepSimulation()
            contacts = p.getContactPoints(self.gripper, self.cube)
            if len(contacts) > 0:
                grasp_stable = True
                break
        
        if grasp_stable:
            self.grasped_object = self.cube
            # Create constraint to hold object with proper offset
            ee_state = p.getLinkState(self.robot, self.ee_index)
            ee_pos = ee_state[0]
            cube_pos, _ = p.getBasePositionAndOrientation(self.cube)
            
            # Calculate proper offset for constraint
            relative_offset = [
                cube_pos[0] - ee_pos[0],
                cube_pos[1] - ee_pos[1],
                cube_pos[2] - ee_pos[2],
            ]
            
            self.grasp_constraint = p.createConstraint(
                self.gripper, -1,
                self.cube, -1,
                p.JOINT_FIXED,
                [0, 0, 0],
                [0, 0, 0],
                relative_offset,
            )
            p.changeConstraint(self.grasp_constraint, maxForce=50)
            print("Object grasped!")
            return True
        return False
    
    def release_object(self):
        """Release the grasped object"""
        if self.grasped_object is not None:
            p.removeConstraint(self.grasp_constraint)
            self.grasped_object = None
            print("Object released")
    
    def test_improved_picking(self):
        """Test improved picking sequence"""
        print("\n=== TESTING IMPROVED PICKING ===")
        
        # Get actual cube position for better targeting
        cube_pos, _ = p.getBasePositionAndOrientation(self.cube)
        
        # Calculate positions based on actual cube location
        pick_position = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.12]  # 12cm above cube
        grasp_position = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.025]  # 2.5cm above cube
        place_position = [0.10, 0.30, 0.20]  # Target location
        
        print(f"Cube at: {cube_pos}")
        print(f"Pick position: {pick_position}")
        print(f"Grasp position: {grasp_position}")
        
        # Step 1: Move above object
        print("\nMoving above object...")
        self.open_gripper()
        self.move_to_position(pick_position, steps=80)
        
        # Check position accuracy
        ee_state = p.getLinkState(self.robot, self.ee_index)
        ee_pos = ee_state[0]
        distance_to_cube = np.linalg.norm(np.array(ee_pos[:2]) - np.array(cube_pos[:2]))
        print(f"Position accuracy: {distance_to_cube:.4f}m from cube center")
        
        # Step 2: Move down to object
        print("Moving down to object...")
        self.move_to_position(grasp_position, steps=60)
        
        # Check final position
        ee_state = p.getLinkState(self.robot, self.ee_index)
        ee_pos = ee_state[0]
        final_distance = np.linalg.norm(np.array(ee_pos) - np.array(cube_pos))
        print(f"Final distance to cube: {final_distance:.4f}m")
        
        # Step 3: Close gripper and grasp
        print("Closing gripper...")
        self.close_gripper()
        
        # Wait for gripper to close and make contact
        print("Waiting for gripper to close...")
        for i in range(80):
            p.stepSimulation()
            if i % 20 == 0:
                contacts = p.getContactPoints(self.gripper, self.cube)
                print(f"  Step {i}: {len(contacts)} contacts")
        
        # Step 4: Check if grasped
        if self.grasp_object():
            print("✅ GRASP SUCCESSFUL!")
            
            # Step 5: Lift object
            print("Lifting object...")
            lift_position = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.15]
            self.move_to_position(lift_position, steps=80)
            
            # Verify object is lifted
            cube_pos_lifted, _ = p.getBasePositionAndOrientation(self.cube)
            lift_height = cube_pos_lifted[2] - cube_pos[2]
            print(f"Object lifted: {lift_height:.4f}m")
            
            # Step 6: Move to place position
            print("Moving to place position...")
            self.move_to_position(place_position, steps=120)
            
            # Step 7: Lower to place
            place_lower = [place_position[0], place_position[1], place_position[2] - 0.08]
            self.move_to_position(place_lower, steps=60)
            
            # Step 8: Release object
            print("Releasing object...")
            self.open_gripper()
            
            # Wait for release
            for _ in range(40):
                p.stepSimulation()
            
            self.release_object()
            
            # Verify object is at place position
            cube_final, _ = p.getBasePositionAndOrientation(self.cube)
            place_accuracy = np.linalg.norm(np.array(cube_final[:2]) - np.array(place_position[:2]))
            print(f"Place accuracy: {place_accuracy:.4f}m from target")
            
            print("✅ PICK AND PLACE COMPLETE!")
            return True
        else:
            print("❌ FAILED TO GRASP OBJECT")
            
            # Try retry
            print("Attempting retry...")
            adjusted_grasp = [cube_pos[0] + 0.01, cube_pos[1] + 0.01, grasp_position[2]]
            self.move_to_position(adjusted_grasp, steps=40)
            
            self.close_gripper()
            for _ in range(60):
                p.stepSimulation()
            
            if self.grasp_object():
                print("✅ RETRY SUCCESSFUL!")
                return True
            else:
                print("❌ RETRY FAILED")
                return False
    
    def run(self):
        """Run test"""
        success = self.test_improved_picking()
        
        if success:
            print("\n🎉 IMPROVED PICKING TEST PASSED!")
        else:
            print("\n💥 IMPROVED PICKING TEST FAILED!")
        
        p.disconnect()

if __name__ == "__main__":
    test = TestImprovedPicking()
    test.run()
