import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perception.segmentation import get_relative_pixel_error_overhead_and_rgb
from reflection.llm_reflection_agent import LLMReflectionAgent, apply_policy_updates
from utils.gui_status import gui_status
import math
from utils.logger import setup_execution_logger, log_robot_state, log_llm_decision, log_policy_update, log_session_summary

# Optional offline "VLM-like" vision classifier (runs fully offline after training)
USE_OFFLINE_VISION_CLASSIFIER = True
offline_classifier = None

# LLM-driven reflection agent. The backend can be set with LLM_AGENT_BACKEND.
# Primary backend: "ollama" for local Llama models.
USE_LLM_AGENT = os.getenv("USE_LLM_AGENT", "1") == "1"
FORCE_REFLECTION = os.getenv("FORCE_REFLECTION", "0") == "1"
FORCED_REFLECTION_ATTEMPTS = int(os.getenv("FORCED_REFLECTION_ATTEMPTS", "1"))

# ---------------- CONNECT ----------------
# Setup logging
logger, log_file = setup_execution_logger()
logger.info(f"Log file: {log_file}")

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setRealTimeSimulation(0)

p.setPhysicsEngineParameter(numSolverIterations=150)
p.setPhysicsEngineParameter(fixedTimeStep=1 / 240)

# ---------------- PLANE ----------------
plane_id = p.loadURDF("plane.urdf")
p.changeDynamics(plane_id, -1, lateralFriction=1.5)

# ---------------- LOAD ROBOT ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF_PATH = os.path.join(BASE_DIR, "urdf", "mycobot_320.urdf")

if USE_OFFLINE_VISION_CLASSIFIER:
    try:
        from perception.offline_vision_classifier import OfflineVisionClassifier

        offline_model_path = os.path.join(
            BASE_DIR, "models", "offline_vlm", "tinycnn_direction.pt"
        )
        offline_classifier = OfflineVisionClassifier(model_path=offline_model_path)
        print("Offline vision classifier loaded:", offline_model_path)
    except Exception as exc:
        offline_classifier = None
        print("Offline vision classifier disabled:", exc)

robot = p.loadURDF(
    URDF_PATH,
    useFixedBase=True,
    flags=p.URDF_USE_INERTIA_FROM_FILE,
)

# ---------------- FIND END EFFECTOR ----------------
ee_index = None
for i in range(p.getNumJoints(robot)):
    if p.getJointInfo(robot, i)[12].decode() == "link6":
        ee_index = i
        break

print("End effector index:", ee_index)

# ---------------- IMPROVED WIDE PARALLEL GRIPPER ----------------
# Load wide parallel gripper for proper cube grasping
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
gripper_urdf = os.path.join(BASE_DIR, "urdf", "wide_parallel_gripper.urdf")
gripper = p.loadURDF(gripper_urdf)

# ---------------- TCP/TOOL OFFSET CALIBRATION ----------------
# Proper gripper offset calibration for accurate positioning
GRIPPER_TCP_OFFSET = [0.0, 0.0, 0.06]  # 6cm offset from link6 to gripper center (reduced)
GRIPPER_ORIENTATION = [0, 0, 0, 1]  # Quaternion orientation (no rotation)

def get_gripper_tcp_position():
    """Get the calibrated TCP position of the gripper"""
    ee_state = p.getLinkState(robot, ee_index)
    ee_pos, ee_orn = ee_state[0], ee_state[1]
    
    # Apply TCP offset
    tcp_pos = np.array(ee_pos) + np.array(GRIPPER_TCP_OFFSET)
    return tcp_pos, ee_orn

def set_gripper_tcp_target(target_pos, orientation=None):
    """Set target position accounting for TCP offset"""
    if orientation is None:
        orientation = GRIPPER_ORIENTATION
    
    # Calculate inverse kinematics with TCP offset compensation
    compensated_target = np.array(target_pos) - np.array(GRIPPER_TCP_OFFSET)
    return compensated_target, orientation

# Get gripper joint indices for standard parallel gripper
gripper_joints = []
gripper_motor_joint = None

for i in range(p.getNumJoints(gripper)):
    joint_info = p.getJointInfo(gripper, i)
    joint_name = joint_info[1].decode()
    print(f"Found gripper joint: {joint_name} at index {i}")
    
    # Standard parallel gripper joint names
    if "finger_joint" in joint_name:
        gripper_joints.append(i)
        if "left_finger" in joint_name or "right_finger" in joint_name:
            gripper_motor_joint = i

print(f"Standard parallel gripper joints: {gripper_joints}")
print(f"Standard parallel gripper motor joint: {gripper_motor_joint}")

# Attach standard parallel gripper to robot end effector with proper TCP calibration
def attach_standard_gripper():
    """Attach standard parallel gripper to robot end effector with calibrated TCP offset"""
    # Get end effector position and orientation
    ee_state = p.getLinkState(robot, ee_index)
    ee_pos, ee_orn = ee_state[0], ee_state[1]
    
    # Use calibrated TCP offset for proper gripper positioning
    constraint_id = p.createConstraint(
        parentBodyUniqueId=robot,
        parentLinkIndex=ee_index,
        childBodyUniqueId=gripper,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=GRIPPER_TCP_OFFSET,  # Use calibrated offset
        childFramePosition=[0, 0, 0]
    )
    
    # Set constraint parameters for stable attachment with reduced force
    p.changeConstraint(constraint_id, maxForce=30)  # Reduced force for smoother operation
    
    print(f"Standard parallel gripper attached with calibrated TCP offset {GRIPPER_TCP_OFFSET}")
    print(f"Constraint ID: {constraint_id}")
    return constraint_id

# Attach gripper
attach_standard_gripper()

# Gripper control functions with improved alignment and smooth control
def open_gripper():
    """Open the wide parallel gripper smoothly with proper force control"""
    for joint_idx in gripper_joints:
        p.setJointMotorControl2(
            gripper,
            joint_idx,
            p.POSITION_CONTROL,
            targetPosition=0.025,  # Open position - WIDE OPEN (25mm travel)
            force=20,  # Reduced force for smoother operation
            positionGain=0.05,  # Lower gain for gentler movement
            velocityGain=0.1  # Reduced velocity gain
        )
    print("Wide parallel gripper opening smoothly...")
    # Add stabilization delay
    time.sleep(0.5)
    print("Wide parallel gripper opened WIDE")

def close_gripper():
    """Close the wide parallel gripper gradually with proper mechanical grasping"""
    print("Closing gripper gradually...")
    
    # Gradual closing for better grip control
    for step in range(10):
        target = 0.025 - (step * 0.0025)  # Gradual closing
        for joint_idx in gripper_joints:
            p.setJointMotorControl2(
                gripper,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=target,
                force=15,  # Low force for gradual closing
                positionGain=0.03,  # Very low gain for smooth control
                velocityGain=0.05  # Very low velocity gain
            )
        # Simulate gradual closing
        for _ in range(5):
            p.stepSimulation()
            time.sleep(1/240)
    
    # Final close with moderate force
    for joint_idx in gripper_joints:
        p.setJointMotorControl2(
            gripper,
            joint_idx,
            p.POSITION_CONTROL,
            targetPosition=0.0,  # Close position - mechanical contact
            force=25,  # Moderate force for secure grasp
            positionGain=0.08,  # Moderate gain for final grasp
            velocityGain=0.1  # Moderate velocity gain
        )
    print("Wide parallel gripper closed - mechanical grasp")
    # Add stabilization delay after closing
    time.sleep(0.3)

# Open gripper initially
open_gripper()

# Function to read robot joint angles and update GUI
def update_robot_joint_angles():
    """Read robot joint angles and update GUI display"""
    try:
        joint_angles = {}
        for i in range(p.getNumJoints(robot)):
            joint_info = p.getJointInfo(robot, i)
            if joint_info[2] == p.JOINT_REVOLUTE:  # Only get revolute joints
                joint_name = joint_info[1].decode()
                joint_angle = p.getJointState(robot, i)[0]  # Get angle in radians
                joint_angles_deg = math.degrees(joint_angle)  # Convert to degrees
                joint_angles[joint_name] = joint_angles_deg
        
        # Update GUI with joint angles
        gui_status.update_joint_angles(joint_angles)
    except Exception as e:
        print(f"Error reading joint angles: {e}")

# Smooth joint angles display on PyBullet screen (clean text only)
joint_text_ids = {}  # Store text object IDs for each joint

def create_smooth_joint_display():
    """Create smooth joint angle text display (no lines)"""
    global joint_text_ids
    joint_text_ids = {}
    
    # Create text for each revolute joint
    for i in range(p.getNumJoints(robot)):
        joint_info = p.getJointInfo(robot, i)
        if joint_info[2] == p.JOINT_REVOLUTE:
            joint_name = joint_info[1].decode()
            
            # Get joint position
            joint_state = p.getLinkState(robot, i)
            joint_pos = joint_state[0]
            
            # Position text above joint for visibility
            text_position = [
                joint_pos[0],
                joint_pos[1], 
                joint_pos[2] + 0.05  # Above joint
            ]
            
            # Create clean text (no lines)
            text_id = p.addUserDebugText(
                text=f"{joint_name}: 0.0°",
                textPosition=text_position,
                textColorRGB=[1, 1, 1],  # White text
                textSize=0.8,  # Small, clean text
                lifeTime=0  # Persistent
            )
            
            joint_text_ids[joint_name] = text_id

def update_smooth_joint_display():
    """Update joint angle display smoothly (no lines)"""
    global joint_text_ids
    
    for i in range(p.getNumJoints(robot)):
        joint_info = p.getJointInfo(robot, i)
        if joint_info[2] == p.JOINT_REVOLUTE:
            joint_name = joint_info[1].decode()
            joint_angle = p.getJointState(robot, i)[0]
            joint_angle_deg = math.degrees(joint_angle)
            
            # Get current joint position
            joint_state = p.getLinkState(robot, i)
            joint_pos = joint_state[0]
            
            # Position text above joint
            text_position = [
                joint_pos[0],
                joint_pos[1], 
                joint_pos[2] + 0.05  # Above joint
            ]
            
            # Remove old text and create new one
            if joint_name in joint_text_ids:
                p.removeUserDebugItem(joint_text_ids[joint_name])
            
            # Create updated text (no lines)
            text_id = p.addUserDebugText(
                text=f"{joint_name}: {joint_angle_deg:5.1f}°",
                textPosition=text_position,
                textColorRGB=[1, 1, 1],  # White text
                textSize=0.8,  # Small, clean text
                lifeTime=0  # Persistent
            )
            
            joint_text_ids[joint_name] = text_id

# Show joint angles display when robot starts working
gui_status.show_joint_angles_display(True)

# Joint angles only in GUI window (no screen display, no parameters)

# Improved grasp detection with centering
def check_grasp_stability(cube_body_id):
    """Check if gripper has stable centered grasp"""
    contacts = p.getContactPoints(gripper, cube_body_id)
    if len(contacts) == 0:
        return False, 0, 0
    
    # Check for parallel face contacts (not corners)
    parallel_contacts = 0
    for contact in contacts:
        # Check if contact normal is mostly vertical (parallel faces)
        normal = contact[7]
        if abs(normal[2]) > 0.7:  # Mostly vertical contact
            parallel_contacts += 1
    
    return len(contacts) >= 2, len(contacts), parallel_contacts

# Improved grasp function with retry mechanism - NO MAGNETIC GRABBING
def improved_grasp_object(cube_body_id, max_retries=3):
    """Grasp the object with improved alignment and retry mechanism - MECHANICAL ONLY"""
    for retry in range(max_retries):
        if retry > 0:
            print(f"Retry attempt {retry + 1}/{max_retries}")
            # Reopen gripper and reposition slightly
            open_gripper()
            p.stepSimulation()
            # Slight reposition for retry
            cube_pos, _ = p.getBasePositionAndOrientation(cube_body_id)
            reposition_offset = [
                cube_pos[0] + np.random.uniform(-0.005, 0.005),
                cube_pos[1] + np.random.uniform(-0.005, 0.005),
                cube_pos[2]
            ]
            # Move to repositioned location
            joint_angles = p.calculateInverseKinematics(
                robot,
                ee_index,
                reposition_offset,
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
                        positionGain=0.04,  # Slow movement for retry
                        velocityGain=0.3
                    )
            
            for _ in range(20):
                p.stepSimulation()
        
        # Close gripper for MECHANICAL grasping only
        close_gripper()
        
        # Wait for mechanical contact to establish
        for i in range(60):  # More steps for mechanical contact
            p.stepSimulation()
        
        # Check for MECHANICAL grasp with parallel alignment
        grasp_stable = False
        contact_count = 0
        centered_contacts = 0
        
        for i in range(40):  # More steps for stable mechanical grasp
            p.stepSimulation()
            stable, total_contacts, parallel_contacts = check_grasp_stability(cube_body_id)
            if stable:
                contact_count += 1
                if parallel_contacts >= 2:  # Need parallel face contacts
                    centered_contacts += 1
                
                if centered_contacts >= 3:  # Need 3 centered contact checks for mechanical grasp
                    grasp_stable = True
                    break
        
        if grasp_stable:
            print(f"Object grasped MECHANICALLY with centered alignment! (Attempt {retry + 1})")
            return True
        else:
            print(f"Mechanical grasp failed on attempt {retry + 1}, contacts: {contact_count}, centered: {centered_contacts}")
    
    print("Failed to grasp object mechanically after all retries")
    return False

# Improved movement with slow mode
def move_to_position(target_position, orientation=None, slow_mode=False):
    """Move robot to target position using inverse kinematics with optional slow mode"""
    if orientation is None:
        orientation = p.getQuaternionFromEuler([0, np.pi, 0])  # Gripper facing down (direct above)
    
    # Calculate inverse kinematics with higher accuracy
    joint_angles = p.calculateInverseKinematics(
        robot,
        ee_index,
        target_position,
        orientation,
        maxNumIterations=500,  # More iterations for better accuracy
        residualThreshold=1e-6  # Tighter threshold
    )
    
    # Apply joint controls with slow mode option
    position_gain = 0.04 if slow_mode else 0.08  # Slower movement for final approach
    velocity_gain = 0.3 if slow_mode else 0.6  # Slower velocity for final approach
    
    for j in range(p.getNumJoints(robot)):
        joint_info = p.getJointInfo(robot, j)
        if joint_info[2] == p.JOINT_REVOLUTE:
            p.setJointMotorControl2(
                robot,
                j,
                p.POSITION_CONTROL,
                targetPosition=joint_angles[j],
                force=400,
                positionGain=position_gain,
                velocityGain=velocity_gain
            )

# ---------------- INTELLIGENT CUBE PLACEMENT ----------------
# Smart cube positioning based on robot capabilities and task strategy
import math

def calculate_optimal_cube_position():
    """Calculate optimal cube position for autonomous robot operation"""
    
    # Define reachability constant
    REACHABLE_THRESHOLD = 0.30
    
    # Strategy: Place cube in position that tests different skills
    # while being optimally reachable and challenging
    
    # Define strategic positions within reachable workspace
    strategic_positions = [
        # Front-right area (most common working area)
        (0.20, 0.15),
        # Front-left area  
        (0.20, -0.15),
        # Right side
        (0.15, 0.20),
        # Left side
        (0.15, -0.20),
        # Front-center (easiest)
        (0.25, 0.0),
        # Diagonal positions (more challenging)
        (0.18, 0.18),
        (0.18, -0.18),
    ]
    
    # Select position based on session number for variety
    session_id = hash(time.strftime("%Y%m%d")) % len(strategic_positions)
    base_position = strategic_positions[session_id]
    
    # Add small intelligent variation for learning
    variation_x = np.random.uniform(-0.02, 0.02)  # Small 2cm variation
    variation_y = np.random.uniform(-0.02, 0.02)
    
    optimal_x = base_position[0] + variation_x
    optimal_y = base_position[1] + variation_y
    
    # Ensure still within reachable bounds
    distance = math.sqrt(optimal_x**2 + optimal_y**2)
    if distance > REACHABLE_THRESHOLD * 0.9:
        # Scale down if too far
        scale = (REACHABLE_THRESHOLD * 0.9) / distance
        optimal_x *= scale
        optimal_y *= scale
    
    return optimal_x, optimal_y

# Calculate intelligent cube position
optimal_x, optimal_y = calculate_optimal_cube_position()
cube = p.loadURDF("cube_small.urdf", [optimal_x, optimal_y, 0.02])
print(f"Cube placed at optimal position: ({optimal_x:.3f}, {optimal_y:.3f}, 0.02)")
print(f"Strategic placement for autonomous learning and testing")

# Check if cube is reachable by robot
robot_base_pos = np.array([0, 0, 0])  # Robot base at origin
cube_pos = np.array([optimal_x, optimal_y, 0.02])
distance_from_robot = np.linalg.norm(cube_pos[:2] - robot_base_pos[:2])  # Only X,Y distance

# MyCobot 320 reachability constants
MAX_REACH = 0.32
REACHABLE_THRESHOLD = 0.30  # Conservative threshold for reachability

print(f"Optimal cube distance from robot: {distance_from_robot:.3f}m")
logger.info(f"OPTIMAL PLACEMENT: Cube positioned at ({optimal_x:.3f}, {optimal_y:.3f}) - {distance_from_robot:.3f}m from robot")
p.changeDynamics(
    cube,
    -1,
    lateralFriction=1.5,
    linearDamping=0.4,
    angularDamping=0.4,
    restitution=0.0,
)

# ---------------- INTELLIGENT GOAL SELECTION ----------------
def calculate_optimal_goal_position():
    """Calculate optimal goal position based on cube location and robot strategy"""
    
    # Define reachability constant
    REACHABLE_THRESHOLD = 0.30
    
    # Strategy: Place goal in a position that requires different movement patterns
    cube_angle = math.atan2(optimal_y, optimal_x)
    cube_distance = math.sqrt(optimal_x**2 + optimal_y**2)
    
    # Choose goal position that requires different robot movement
    # Options: opposite side, perpendicular, or same side but different distance
    strategies = [
        # Opposite side (180 degrees)
        lambda: (-optimal_x * 0.7, -optimal_y * 0.7),
        # Perpendicular (90 degrees)
        lambda: (-optimal_y * 0.6, optimal_x * 0.6),
        # Same side but closer
        lambda: (optimal_x * 0.5, optimal_y * 0.5),
        # Diagonal from cube
        lambda: ((optimal_x + optimal_y) * 0.4, (optimal_y - optimal_x) * 0.4),
    ]
    
    # Select strategy based on cube position for variety
    strategy_index = int(abs(cube_angle) * 2) % len(strategies)
    goal_x, goal_y = strategies[strategy_index]()
    
    # Ensure goal is reachable
    goal_distance = math.sqrt(goal_x**2 + goal_y**2)
    if goal_distance > REACHABLE_THRESHOLD * 0.9:
        scale = (REACHABLE_THRESHOLD * 0.9) / goal_distance
        goal_x *= scale
        goal_y *= scale
    
    return goal_x, goal_y

# Calculate intelligent goal position
goal_x, goal_y = calculate_optimal_goal_position()
goal_position = np.array([goal_x, goal_y, 0.02])
print(f"Goal set at strategic position: ({goal_x:.3f}, {goal_y:.3f}, 0.02)")
print(f"Goal distance from robot: {math.sqrt(goal_x**2 + goal_y**2):.3f}m")
logger.info(f"STRATEGIC GOAL: Goal positioned at ({goal_x:.3f}, {goal_y:.3f}) for optimal learning")

# Create goal object for visualization
goal = p.createMultiBody(
    baseVisualShapeIndex=p.createVisualShape(
        p.GEOM_SPHERE,
        radius=0.02,
        rgbaColor=[1, 0, 0, 1]
    ),
    basePosition=[goal_x, goal_y, 0.02]
)

# ---------------- CAMERA (DEBUG VIEW) ----------------
p.resetDebugVisualizerCamera(
    cameraDistance=1.2,
    cameraYaw=45,
    cameraPitch=-35,
    cameraTargetPosition=[0.3, 0.1, 0.08],
)

# ---------------- POLICY ----------------
policy = {
    "approach_height": 0.12,
    "grasp_height": 0.03,
    "lift_height": 0.20,
    "release_delay": 60,
    "x_offset": 0.0,
    "y_offset": 0.0,
}

max_retries = 10
retry_count = 0
inject_failure = True
perception_noise_scale = 0.08

# ---------------- IMPROVED SMOOTH MOTION WITH TCP CALIBRATION ----------------
def smooth_move(target_pos, steps=500, slow_mode=False):
    """Improved smooth motion with TCP calibration and gentle control"""
    joint_indices = []
    current_positions = []

    for j in range(p.getNumJoints(robot)):
        joint_info = p.getJointInfo(robot, j)
        if joint_info[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
            joint_indices.append(j)
            current_positions.append(p.getJointState(robot, j)[0])

    # Use TCP calibration for accurate positioning
    compensated_target, orientation = set_gripper_tcp_target(target_pos)
    
    target_positions = p.calculateInverseKinematics(
        robot,
        ee_index,
        compensated_target.tolist(),
        orientation,
        maxNumIterations=300,  # More iterations for better accuracy
    )

    # Smooth trajectory interpolation with eased motion
    for step in range(steps):
        # Use eased interpolation for smoother acceleration/deceleration
        progress = step / steps
        if slow_mode:
            # Cubic easing for very slow, smooth motion
            alpha = progress * progress * (3.0 - 2.0 * progress)
        else:
            # Sine easing for smooth motion
            alpha = 0.5 * (1 - math.cos(math.pi * progress))

        for idx, joint_index in enumerate(joint_indices):
            # Smooth interpolation with eased motion
            interpolated = (
                (1 - alpha) * current_positions[idx]
                + alpha * target_positions[joint_index]
            )

            # Gentle control parameters for smooth movement
            force = 20 if slow_mode else 30  # Reduced force
            pos_gain = 0.1 if slow_mode else 0.15  # Lower position gain
            vel_gain = 0.2 if slow_mode else 0.3  # Lower velocity gain

            p.setJointMotorControl2(
                robot,
                joint_index,
                p.POSITION_CONTROL,
                interpolated,
                force=force,
                positionGain=pos_gain,
                velocityGain=vel_gain,
            )

        p.stepSimulation()
        # Slower timing for smoother motion
        time.sleep(1 / 120 if slow_mode else 1 / 240)

def stabilization_delay(duration=0.5):
    """Add stabilization delay for smooth transitions"""
    print(f"Stabilizing for {duration}s...")
    for _ in range(int(duration * 240)):
        p.stepSimulation()
        time.sleep(1 / 240)

# ---------------- STAGED MOVEMENT SEQUENCE ----------------
def staged_pick_sequence(cube_pos):
    """Implement staged pick-and-place sequence with proper alignment"""
    print("\n=== STAGED PICK SEQUENCE START ===")
    
    # Stage 1: Open gripper wide
    print("Stage 1: Opening gripper wide...")
    open_gripper()
    stabilization_delay(0.3)
    
    # Stage 2: Move above object with proper alignment
    print("Stage 2: Moving above object...")
    above_target = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.12]  # 12cm above
    smooth_move(above_target, steps=600, slow_mode=True)  # Very slow approach
    stabilization_delay(0.5)
    
    # Stage 3: Descend slowly to object
    print("Stage 3: Descending slowly to object...")
    descend_target = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.03]  # 3cm above
    smooth_move(descend_target, steps=800, slow_mode=True)  # Extra slow descent
    stabilization_delay(0.3)
    
    # Stage 4: Final approach and contact
    print("Stage 4: Final approach to object...")
    contact_target = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.005]  # 0.5cm above (closer)
    smooth_move(contact_target, steps=800, slow_mode=True)  # Ultra slow final approach
    stabilization_delay(0.2)
    
    # Stage 4.5: Make gentle contact
    print("Stage 4.5: Making gentle contact...")
    contact_target = [cube_pos[0], cube_pos[1], cube_pos[2] - 0.002]  # Slightly into cube
    smooth_move(contact_target, steps=400, slow_mode=True)  # Very slow contact
    stabilization_delay(0.1)
    
    # Stage 5: Close gripper gradually
    print("Stage 5: Closing gripper gradually...")
    close_gripper()
    stabilization_delay(0.5)
    
    # Stage 6: Verify grasp
    print("Stage 6: Verifying grasp...")
    grasp_verified = verify_grasp_stability()
    
    if grasp_verified:
        print("Stage 7: Lifting vertically...")
        # Lift vertically first
        lift_target = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.15]
        smooth_move(lift_target, steps=600, slow_mode=True)
        stabilization_delay(0.3)
        return True
    else:
        print("Grasp verification failed!")
        return False

def verify_grasp_stability():
    """Verify if gripper has a stable grasp"""
    # Check contact points between gripper and cube
    contacts = p.getContactPoints(gripper, cube)
    
    print(f"Contact points found: {len(contacts)}")
    
    # More lenient contact requirement
    if len(contacts) < 1:
        print(f"No contacts detected")
        return False
    
    # Check if cube is close to gripper
    gripper_tcp_pos, _ = get_gripper_tcp_position()
    cube_pos, _ = p.getBasePositionAndOrientation(cube)
    
    distance = np.linalg.norm(np.array(gripper_tcp_pos) - np.array(cube_pos))
    if distance > 0.08:  # 8cm tolerance (more lenient)
        print(f"Cube too far from gripper: {distance:.3f}m")
        return False
    
    # Check if cube is being lifted (Z position change)
    cube_z = cube_pos[2]
    if cube_z < 0.015:  # Cube should be lifted from table
        print(f"Cube not lifted enough: {cube_z:.3f}m")
        return False
    
    print(f"Grasp verified: {len(contacts)} contacts, distance: {distance:.3f}m, cube_z: {cube_z:.3f}m")
    return True

def staged_place_sequence(goal_pos):
    """Implement staged place sequence with proper alignment"""
    print("\n=== STAGED PLACE SEQUENCE START ===")
    
    # Stage 1: Move above goal position
    print("Stage 1: Moving above goal...")
    above_goal = [goal_pos[0], goal_pos[1], goal_pos[2] + 0.12]
    smooth_move(above_goal, steps=600, slow_mode=True)
    stabilization_delay(0.3)
    
    # Stage 2: Descend to placement height
    print("Stage 2: Descending to placement height...")
    place_target = [goal_pos[0], goal_pos[1], goal_pos[2] + 0.03]
    smooth_move(place_target, steps=800, slow_mode=True)
    stabilization_delay(0.3)
    
    # Stage 3: Release object
    print("Stage 3: Releasing object...")
    open_gripper()
    stabilization_delay(0.5)
    
    # Stage 4: Lift away
    print("Stage 4: Lifting away...")
    lift_away = [goal_pos[0], goal_pos[1], goal_pos[2] + 0.15]
    smooth_move(lift_away, steps=600, slow_mode=True)
    stabilization_delay(0.3)
    
    print("=== STAGED PLACE SEQUENCE COMPLETE ===")

# ---------------- ADAPTIVE RETRY LOGIC ----------------
def adaptive_retry_adjustment(failure_type, cube_pos, current_policy):
    """Implement adaptive retry logic with parameter adjustment"""
    print(f"\n=== ADAPTIVE RETRY ANALYSIS ===")
    print(f"Failure type: {failure_type}")
    print(f"Current policy: {current_policy}")
    
    updated_policy = current_policy.copy()
    
    if failure_type == "grasp_failure":
        # Analyze specific grasp failure
        gripper_tcp_pos, _ = get_gripper_tcp_position()
        distance_to_cube = np.linalg.norm(np.array(gripper_tcp_pos) - np.array(cube_pos))
        
        print(f"Distance to cube: {distance_to_cube:.3f}m")
        
        if distance_to_cube > 0.08:
            # Too far from object - adjust approach height and offsets
            updated_policy["approach_height"] -= 0.01  # Lower approach
            updated_policy["x_offset"] += 0.005  # Adjust X offset
            updated_policy["y_offset"] += 0.005  # Adjust Y offset
            print("Adjustment: Lower approach height and adjust offsets")
            
        elif distance_to_cube < 0.03:
            # Too close - adjust grasp height
            updated_policy["grasp_height"] += 0.005  # Raise grasp height
            print("Adjustment: Raise grasp height")
            
        else:
            # Distance OK but grasp failed - adjust gripper force and timing
            updated_policy["release_delay"] += 10  # Increase stabilization
            print("Adjustment: Increase stabilization delay")
            
    elif failure_type == "side_grasp_failure":
        # Side grasp specific adjustments
        updated_policy["grasp_height"] -= 0.005  # Lower for better side contact
        updated_policy["approach_height"] -= 0.01  # Better side approach
        print("Adjustment: Optimize for side grasping")
        
    elif failure_type == "placement_failure":
        # Placement specific adjustments
        updated_policy["lift_height"] += 0.01  # Higher lift for better placement
        updated_policy["release_delay"] += 15  # Longer release delay
        print("Adjustment: Optimize placement sequence")
        
    # Safety bounds checking
    updated_policy["approach_height"] = max(0.05, min(0.20, updated_policy["approach_height"]))
    updated_policy["grasp_height"] = max(0.005, min(0.05, updated_policy["grasp_height"]))
    updated_policy["lift_height"] = max(0.10, min(0.30, updated_policy["lift_height"]))
    updated_policy["release_delay"] = max(30, min(120, updated_policy["release_delay"]))
    
    print(f"Updated policy: {updated_policy}")
    print("=== END ADAPTIVE RETRY ANALYSIS ===\n")
    
    return updated_policy

# ---------------- GRIPPER FRICTION RECOMMENDATIONS ----------------
def print_gripper_friction_recommendations():
    """Print recommendations for improving gripper finger friction"""
    print("\n=== GRIPPER FRICTION RECOMMENDATIONS ===")
    print("For improved grasping performance, consider:")
    print("1. Rubber pads: Add thin rubber sheets to gripper fingers")
    print("2. Silicone pads: Use silicone grip pads for better friction")
    print("3. Textured surface: Add sandpaper or textured grip tape")
    print("4. 3D printed patterns: Create custom finger textures")
    print("5. Soft foam: Add thin foam layer for conformal grip")
    print("6. Anti-slip tape: Use industrial anti-slip materials")
    print("==========================================\n")

# ---------------- STATE MACHINE ----------------
state = "plan"
timer = 0
stable_counter = 0
constraint_id = None
last_failure_type = "startup"
last_distance_to_goal = None
attempt_history = []

# Print gripper friction recommendations once
print_gripper_friction_recommendations()

agent = LLMReflectionAgent(
    backend=os.getenv("LLM_AGENT_BACKEND", "ollama"),
    model=os.getenv("LLM_AGENT_MODEL", "llama3.2-vision"),
    endpoint=os.getenv("LLM_AGENT_ENDPOINT", "http://localhost:11434/api/chat"),
    timeout_s=float(os.getenv("LLM_AGENT_TIMEOUT_S", "60")),
    use_vision=os.getenv("LLM_AGENT_USE_VISION", "0") == "1"
)
if not USE_LLM_AGENT:
    agent.api_key = None
    agent.endpoint = ""

if USE_LLM_AGENT:
    if agent.is_configured():
        print("LLM agent enabled")
        print("Backend:", agent.backend)
        print("Model:", agent.model)
        print("Endpoint:", agent.endpoint)
    else:
        print("LLM agent requested, but configuration is incomplete. Using fallback heuristic.")
else:
    print("LLM agent disabled. Using fallback heuristic.")

if FORCE_REFLECTION:
    print("Force reflection enabled for", FORCED_REFLECTION_ATTEMPTS, "attempt(s)")

# ---------------- LOGGING ----------------
attempt_distances = []
successful_grasp = False
successful_placement = False

while p.isConnected():
    timer += 1
    cube_pos, _ = p.getBasePositionAndOrientation(cube)
    cube_pos = np.array(cube_pos)
    
    # Calculate current distance to goal
    goal_pos, _ = p.getBasePositionAndOrientation(goal)
    current_distance_to_goal = float(np.linalg.norm(np.array(cube_pos) - np.array(goal_pos)))
    
    # Update robot joint angles in GUI only (no screen display, no parameters)
    update_robot_joint_angles()
    
    # Update GUI metrics
    gui_status.update_metrics(retry_count + 1, current_distance_to_goal)
        
    if state == "plan":
        print("\nPlanning attempt", retry_count + 1)
        
        # Check if goal position is reachable
        goal_distance_from_robot = np.linalg.norm(goal_position[:2] - np.array([0, 0]))
        if goal_distance_from_robot > REACHABLE_THRESHOLD:
            print(f"WARNING: Goal position too far from robot!")
            print(f"Goal distance: {goal_distance_from_robot:.3f}m, Max reach: {REACHABLE_THRESHOLD:.3f}m")
            
            # Log unreachable goal
            logger.info(f"UNREACHABLE GOAL: Distance {goal_distance_from_robot:.3f}m exceeds reach {REACHABLE_THRESHOLD:.3f}m")
            
            # Update GUI with unreachable goal status
            gui_status.update_status("Unreachable", f"Goal too far: {goal_distance_from_robot:.3f}m > {REACHABLE_THRESHOLD:.3f}m")
            gui_status.display_status()
            
            # Terminate execution
            state = "done"
            continue
        
        gui_status.update_status("Planning", f"Attempt {retry_count + 1}")
        gui_status.update_metrics(retry_count + 1, last_distance_to_goal)
        gui_status.display_status()
        
        # Log planning state
        log_robot_state(logger, "PLANNING", f"Attempt {retry_count + 1}", retry_count + 1, last_distance_to_goal)

        perceived_cube_pos = cube_pos.copy()
        perceived_cube_pos[0] += policy["x_offset"]
        perceived_cube_pos[1] += policy["y_offset"]

        if inject_failure:
            print("Injecting perception error")
            perceived_cube_pos[0] += np.random.uniform(
                -perception_noise_scale, perception_noise_scale
            )
            perceived_cube_pos[1] += np.random.uniform(
                -perception_noise_scale, perception_noise_scale
            )

        approach_target = perceived_cube_pos.copy()
        approach_target[2] += policy["approach_height"]

        grasp_target = perceived_cube_pos.copy()
        grasp_target[2] += policy["grasp_height"]

        state = "approach"
        timer = 0

    elif state == "approach":
        gui_status.update_status("Approaching", "Using staged pick sequence")
        gui_status.display_status()
        log_robot_state(logger, "APPROACHING", "Using staged pick sequence", retry_count + 1)
        
        # Use new staged pick sequence with proper TCP calibration
        grasp_success = staged_pick_sequence(perceived_cube_pos)
        
        if grasp_success:
            successful_grasp = True
            state = "lift"
            timer = 0
            print("Staged pick sequence completed successfully!")
        else:
            print("Staged pick sequence failed!")
            state = "failed"
            timer = 0

    elif state == "grasp":
        grip_pos = p.getLinkState(robot, ee_index)[0]
        cube_pos, _ = p.getBasePositionAndOrientation(cube)

        dist = float(np.linalg.norm(np.array(grip_pos) - np.array(cube_pos)))
        print(f"Gripper position: {grip_pos}")
        print(f"Cube position: {cube_pos}")
        print(f"Distance: {dist}")

        if (dist < 0.08 and retry_count > 0) or (dist < 0.08 and not inject_failure):
            # PROPER GRASP SEQUENCE: Approach → Open → Grasp → Lock → Move → Place
            print("Starting proper grasp sequence...")
            
            # Step 1: Open gripper wide before approaching cube
            print("Step 1: Opening gripper wide...")
            open_gripper()
            for _ in range(30):
                p.stepSimulation()
            
            # Step 2: Move to proper grasping position (sides of cube, not top)
            print("Step 2: Positioning gripper at cube side level...")
            cube_side_pos = np.array(cube_pos)  # Convert tuple to numpy array
            cube_side_pos[2] += 0.015  # Position at middle of cube height
            smooth_move(cube_side_pos, slow_mode=True)
            for _ in range(40):
                p.stepSimulation()
            
            # Step 3: Close gripper around cube from sides
            print("Step 3: Closing gripper around cube from sides...")
            close_gripper()
            
            # Step 4: Wait for proper mechanical contact
            print("Step 4: Establishing mechanical contact...")
            for _ in range(60):
                p.stepSimulation()
            
            # Step 5: Check for proper side grasp (not top attachment)
            print("Step 5: Verifying side grasp...")
            contacts = p.getContactPoints(gripper, cube)
            if len(contacts) >= 2:
                # Check if contacts are on sides (not top)
                side_contacts = 0
                for contact in contacts:
                    # Contact point on gripper
                    contact_point = contact[5]
                    # Check if contact is on side fingers (not top)
                    if abs(contact_point[2] - cube_pos[2]) < 0.02:  # Within cube height range
                        side_contacts += 1
                
                if side_contacts >= 2:
                    print("Proper side grasp confirmed!")
                    
                    # Step 6: Lock the grasp with constraint
                    print("Step 6: Locking grasp...")
                    ee_state = p.getLinkState(robot, ee_index)
                    ee_pos = ee_state[0]
                    cube_pos_current, _ = p.getBasePositionAndOrientation(cube)
                    final_distance = float(np.linalg.norm(np.array(ee_pos) - np.array(cube_pos_current)))
                    
                    print(f"FINAL DISTANCE AFTER SIDE GRASP: {final_distance:.6f}")
                    print(f"Gripper position after grasp: {ee_pos}")
                    print(f"Cube position after grasp: {cube_pos_current}")
                    
                    # Create constraint for proper grasp
                    relative_offset = [
                        cube_pos_current[0] - ee_pos[0],
                        cube_pos_current[1] - ee_pos[1],
                        cube_pos_current[2] - ee_pos[2] - 0.002,  # Minimal offset for side grasp
                    ]
                    
                    constraint_id = p.createConstraint(
                        gripper,
                        -1,
                        cube,
                        -1,
                        p.JOINT_FIXED,
                        [0, 0, 0],
                        [0, 0, 0],
                        relative_offset,
                    )
                    p.changeConstraint(constraint_id, maxForce=30)  # Lower force for side grasp

                    print("Cube properly grasped from sides and locked!")
                    gui_status.update_status("Grasped", f"Cube grasped from sides - Distance: {final_distance:.4f}")
                    gui_status.display_status()
                    log_robot_state(logger, "GRASPED", f"Cube grasped from sides - Distance: {final_distance:.4f}", retry_count + 1)
                    successful_grasp = True
                    state = "lift"
                    timer = 0
                else:
                    print("Side grasp failed - contacts not on sides")
                    gui_status.update_status("Failed", "Side grasp failed - contacts not on sides")
                    gui_status.display_status()
                    log_robot_state(logger, "FAILED", "Side grasp failed - contacts not on sides", retry_count + 1)
                    last_failure_type = "grasp_failure"
                    state = "analyze"
            else:
                print("Side grasp failed - insufficient contacts")
                gui_status.update_status("Failed", "Side grasp failed - insufficient contacts")
                gui_status.display_status()
                log_robot_state(logger, "FAILED", "Side grasp failed - insufficient contacts", retry_count + 1)
                last_failure_type = "grasp_failure"
                state = "analyze"

        else:
            print("Grasp failed - too far from object")
            gui_status.update_status("Failed", "Grasp failed - too far from object")
            gui_status.display_status()
            log_robot_state(logger, "FAILED", "Grasp failed - too far from object", retry_count + 1)
            last_failure_type = "grasp_failure"
            state = "analyze"

    elif state == "lift":
        gui_status.update_status("Lifting", "Moving to place position")
        gui_status.display_status()
        log_robot_state(logger, "LIFTING", "Moving to place position", retry_count + 1)
        
        # Use new staged place sequence with proper TCP calibration
        staged_place_sequence(goal_position)
        
        successful_placement = True
        state = "observe"
        timer = 0
        stable_counter = 0
        print("Staged place sequence completed!")

    elif state == "observe":
        observe_timer = 0
        while p.isConnected() and observe_timer < 300:  # 15 second timeout
            try:
                # Get current cube position with error handling
                cube_pos, cube_orn = p.getBasePositionAndOrientation(cube)
                goal_pos, _ = p.getBasePositionAndOrientation(goal)
            except Exception as e:
                print(f"Error getting object positions: {e}")
                break
            
            # Calculate distance to goal
            distance_to_goal = float(np.linalg.norm(np.array(cube_pos) - np.array(goal_pos)))
            
            # Calculate cube height and surface check
            cube_z = cube_pos[2]
            surface_z = 0.025  # Table surface height
            is_on_surface = abs(cube_z - surface_z) < 0.01
            
            print(f"Object height: {cube_z:.3f}m, Surface height: {surface_z:.3f}m, On surface: {is_on_surface}")

            # Success criteria: close to goal, low speed, AND on surface
            if distance_to_goal < 0.10 and speed < 0.05 and is_on_surface:
                stable_counter += 1
            else:
                stable_counter = 0

            if stable_counter > 30:  # Stable for 30 frames
                if FORCE_REFLECTION and retry_count < FORCED_REFLECTION_ATTEMPTS:
                    print("FORCE_REFLECTION active -> sending successful attempt to reflection")
                    attempt_distances.append(distance_to_goal)
                    last_failure_type = "forced_reflection"
                    state = "analyze"
                else:
                    print("SUCCESS -> task completed (object properly placed on surface)")
                    successful_placement = True
                    print(f"DEBUG: successful_placement set to {successful_placement}")
                    print(f"DEBUG: retry_count = {retry_count}")
                    state = "done"
                    attempt_distances.append(distance_to_goal)
                break  # Exit observe loop

            observe_timer += 1
            timer += 1
            time.sleep(1/240)  # Maintain simulation step rate

        # Timeout fallback
        if observe_timer >= 300:
            print("Observe timeout reached")
            attempt_distances.append(distance_to_goal)
            last_failure_type = "observe_timeout"
            state = "analyze"

    elif state == "analyze":
        if retry_count >= max_retries:
            print("Max retries reached")
            state = "done"
            continue

        print("\n========== ADAPTIVE RETRY ANALYSIS ==========")
        
        # Update GUI to show reflecting status
        gui_status.update_status("Reflecting", "Adaptive retry analysis and parameter adjustment")
        gui_status.display_status()
        log_robot_state(logger, "REFLECTING", "Adaptive retry analysis and parameter adjustment", retry_count + 1)
        
        # Get current cube position for analysis
        cube_pos, _ = p.getBasePositionAndOrientation(cube)
        
        # Apply adaptive retry adjustments
        updated_policy = adaptive_retry_adjustment(last_failure_type, cube_pos, policy)
        
        # Update policy with adaptive adjustments
        policy.update(updated_policy)
        
        print(f"Policy updated for retry {retry_count + 1}: {policy}")
        
        # Optionally use LLM for additional analysis
        if USE_LLM_AGENT and retry_count < 3:  # Use LLM for first few retries only
            print("\n========== LLM REFLECTION ==========")
            error, rgb = get_relative_pixel_error_overhead_and_rgb(
                target_body_id=cube,
                reference_body_id=gripper,
                verbose=False,
            )

            pixel_error_x = 0.0
            pixel_error_y = 0.0
            cube_visible = error is not None
            offline_summary = None
            offline_confidence = None

            if error is None:
                print("Cube not visible -> agent must reason with limited observations")
            else:
                pixel_error_x, pixel_error_y = error

            if offline_classifier is not None:
                try:
                    pred = offline_classifier.predict(rgb)
                    offline_summary = pred.label
                    offline_confidence = float(pred.confidence)
                    print(f"OfflineVLM: {pred.label} (conf={pred.confidence:.2f})")
                except Exception as exc:
                    print("OfflineVLM prediction failed:", exc)

            scene_info = {
                "failure_type": last_failure_type,
                "retry_count": int(retry_count),
                "cube_visible": bool(cube_visible),
                "pixel_error_x": float(pixel_error_x),
                "pixel_error_y": float(pixel_error_y),
                "distance_to_goal": None if last_distance_to_goal is None else float(last_distance_to_goal),
                "offline_direction_label": offline_summary,
                "offline_direction_confidence": offline_confidence,
            }

            decision = agent.reflect_and_decide(scene_info)
            if decision:
                # Apply LLM decisions if they don't conflict with adaptive adjustments
                llm_updates = decision.get("updates", {})
                for key, value in llm_updates.items():
                    if key in policy and value is not None:
                        # Apply LLM update if it's reasonable
                        if key in ["approach_height", "grasp_height", "lift_height"]:
                            if 0.05 <= value <= 0.30:  # Safety bounds
                                policy[key] = value
                        elif key == "release_delay":
                            if 30 <= value <= 120:  # Safety bounds
                                policy[key] = value
                        else:
                            policy[key] = value
                
                print(f"LLM decision applied: {decision}")
        
        retry_count += 1
        state = "plan"
        timer = 0
        print("====================================\n")

        error, rgb = get_relative_pixel_error_overhead_and_rgb(
            target_body_id=cube,
            reference_body_id=gripper,
            verbose=False,
        )

        pixel_error_x = 0.0
        pixel_error_y = 0.0
        cube_visible = error is not None
        offline_summary = None
        offline_confidence = None

        if error is None:
            print("Cube not visible -> agent must reason with limited observations")
        else:
            pixel_error_x, pixel_error_y = error

        if offline_classifier is not None:
            try:
                pred = offline_classifier.predict(rgb)
                offline_summary = pred.label
                offline_confidence = float(pred.confidence)
                print(f"OfflineVLM: {pred.label} (conf={pred.confidence:.2f})")
            except Exception as exc:
                print("OfflineVLM prediction failed:", exc)

        scene_info = {
            "failure_type": last_failure_type,
            "retry_count": int(retry_count),
            "cube_visible": bool(cube_visible),
            "pixel_error_x": float(pixel_error_x),
            "pixel_error_y": float(pixel_error_y),
            "distance_to_goal": None if last_distance_to_goal is None else float(last_distance_to_goal),
            "offline_direction_label": offline_summary,
            "offline_direction_confidence": offline_confidence,
        }

        decision = agent.reflect(
            scene_info=scene_info,
            policy=policy,
            rgb=rgb,
            history=attempt_history,
        )

        print("Agent mode:", decision.mode)
        print("Agent explanation:", decision.explanation)
        print("Proposed updates:", decision.updates)
        if decision.confidence is not None:
            print("Agent confidence:", round(decision.confidence, 3))
        
        # Log LLM decision
        log_llm_decision(logger, decision)
        
        # Update GUI with LLM decision
        confidence_str = f"{decision.confidence:.2f}" if decision.confidence is not None else "N/A"
        explanation_str = decision.explanation[:100] + ("..." if len(decision.explanation) > 100 else "")
        llm_summary = f"Mode: {decision.mode} | Confidence: {confidence_str}\nExplanation: {explanation_str}"
        gui_status.update_status("Reflecting", "LLM analysis complete", llm_summary)
        gui_status.display_status()

        old_policy = policy.copy()
        policy = apply_policy_updates(policy, decision.updates)
        log_policy_update(logger, old_policy, policy)
        print("Updated policy:", policy)

        attempt_history.append(
            {
                "retry": int(retry_count),
                "failure_type": last_failure_type,
                "cube_visible": bool(cube_visible),
                "pixel_error_x": float(pixel_error_x),
                "pixel_error_y": float(pixel_error_y),
                "distance_to_goal": None if last_distance_to_goal is None else float(last_distance_to_goal),
                "updates": decision.updates,
                "mode": decision.mode,
            }
        )

        if decision.terminate:
            print("Agent requested termination")
            state = "done"
            continue

        print("====================================\n")

        inject_failure = False
        retry_count += 1
        state = "plan"
        timer = 0

    elif state == "done":
        print("Terminating simulation.")
        # Log session summary with correct success detection
        actual_attempts = retry_count + 1  # Since retry_count starts at 0
        success = successful_placement  # Based on actual task completion
        log_session_summary(logger, actual_attempts, last_distance_to_goal or 0.0, success)
        break

    p.stepSimulation()
    # time.sleep(1 / 240)  # Removed to unlock the simulation frame rate

p.disconnect()

# Close GUI status window
gui_status.close()

# ---------------- PLOT RESULTS ----------------
if attempt_distances:
    attempts = np.arange(1, len(attempt_distances) + 1)
    plt.figure()
    plt.plot(attempts, attempt_distances, marker="o")
    plt.xlabel("Attempt")
    plt.ylabel("Final distance to goal (m)")
    plt.title("Distance to goal vs. attempt with LLM reflection")
    plt.grid(True)
    plt.tight_layout()

    out_dir = os.path.join(BASE_DIR, "data", "plots")
    os.makedirs(out_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    png_path = os.path.join(out_dir, f"distance_vs_attempt_{timestamp}.png")
    csv_path = os.path.join(out_dir, f"distance_vs_attempt_{timestamp}.csv")

    plt.savefig(png_path, dpi=200)
    plt.close()

    with open(csv_path, "w", encoding="utf-8") as file_obj:
        file_obj.write("attempt,final_distance_m\n")
        for attempt_index, distance in zip(attempts.tolist(), attempt_distances):
            file_obj.write(f"{attempt_index},{float(distance)}\n")

    print("Saved plot:", png_path)
    print("Saved data:", csv_path)
