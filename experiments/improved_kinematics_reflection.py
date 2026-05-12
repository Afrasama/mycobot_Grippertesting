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

# Attach standard parallel gripper to robot end effector
def attach_standard_gripper():
    """Attach standard parallel gripper to robot end effector"""
    # Get end effector position and orientation
    ee_state = p.getLinkState(robot, ee_index)
    ee_pos = ee_state[0]
    ee_orn = ee_state[1]
    
    # Position gripper at end effector with proper orientation
    gripper_offset = [0, 0, 0.02]  # 2cm forward offset for standard gripper
    gripper_orn_offset = p.getQuaternionFromEuler([0, 0, 0])
    
    # Transform gripper position
    gripper_pos, gripper_orn = p.multiplyTransforms(
        ee_pos, ee_orn,
        gripper_offset, gripper_orn_offset
    )
    
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
    p.changeConstraint(constraint, maxForce=300)
    
    print("Standard parallel gripper attached to robot end effector")

# Attach gripper
attach_standard_gripper()

# Gripper control functions with improved alignment
def open_gripper():
    """Open the wide parallel gripper as wide as possible"""
    for joint_idx in gripper_joints:
        p.setJointMotorControl2(
            gripper,
            joint_idx,
            p.POSITION_CONTROL,
            targetPosition=0.025,  # Open position - WIDE OPEN (25mm travel)
            force=50,
            positionGain=0.1
        )
    print("Wide parallel gripper opened WIDE")

def close_gripper():
    """Close the wide parallel gripper with proper mechanical grasping"""
    for joint_idx in gripper_joints:
        p.setJointMotorControl2(
            gripper,
            joint_idx,
            p.POSITION_CONTROL,
            targetPosition=0.0,  # Close position - mechanical contact (no magnetic)
            force=30,  # Moderate force for mechanical grasp
            positionGain=0.15  # Moderate gain for smooth control
        )
    print("Wide parallel gripper closed - mechanical grasp")

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

# ---------------- IMPROVED SMOOTH MOTION ----------------
def smooth_move(target_pos, steps=300, slow_mode=False):
    """Improved smooth motion with optional slow mode for final approach"""
    joint_indices = []
    current_positions = []

    for j in range(p.getNumJoints(robot)):
        joint_info = p.getJointInfo(robot, j)
        if joint_info[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
            joint_indices.append(j)
            current_positions.append(p.getJointState(robot, j)[0])

    target_positions = p.calculateInverseKinematics(
        robot,
        ee_index,
        target_pos.tolist(),
        maxNumIterations=200,
    )

    for step in range(steps):
        alpha = step / steps

        for idx, joint_index in enumerate(joint_indices):
            interpolated = (
                (1 - alpha) * current_positions[idx]
                + alpha * target_positions[joint_index]
            )

            p.setJointMotorControl2(
                robot,
                joint_index,
                p.POSITION_CONTROL,
                interpolated,
                force=50,  # Reduced force for gentler movement
                positionGain=0.3,  # Softer position control
                velocityGain=0.5,  # Reduced velocity gain
            )

        p.stepSimulation()
        time.sleep(1 / 240)  # Restore proper timing for smooth motion

# ---------------- STATE MACHINE ----------------
state = "plan"
timer = 0
stable_counter = 0
constraint_id = None
last_failure_type = "startup"
last_distance_to_goal = None
attempt_history = []

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
        gui_status.update_status("Approaching", "Moving to side approach position")
        gui_status.display_status()
        log_robot_state(logger, "APPROACHING", "Moving to side approach position", retry_count + 1)
        
        # Calculate side approach position (not above)
        cube_pos = perceived_cube_pos
        side_approach_target = cube_pos.copy()
        side_approach_target[0] += 0.08  # Approach from side (8cm offset)
        side_approach_target[2] += 0.03  # Slightly above cube middle
        
        smooth_move(side_approach_target, slow_mode=False)  # Normal speed for approach
        state = "descend"
        timer = 0

    elif state == "descend":
        gui_status.update_status("Descending", "Moving to cube side for grasping")
        gui_status.display_status()
        log_robot_state(logger, "DESCENDING", "Moving to cube side for grasping", retry_count + 1)
        
        # Move to cube side position (not above)
        cube_pos = perceived_cube_pos
        side_grasp_target = cube_pos.copy()
        side_grasp_target[2] += 0.015  # Position at middle of cube height
        
        smooth_move(side_grasp_target, slow_mode=True)  # Slow movement for final approach
        state = "grasp"
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
        lift_target = cube_pos.copy()
        lift_target[2] += policy["lift_height"]
        smooth_move(lift_target, slow_mode=True)  # Slow lift for stability
        state = "place_above"
        timer = 0

    elif state == "place_above":
        above_goal = goal_position.copy()
        above_goal[2] += policy["approach_height"]
        smooth_move(above_goal, slow_mode=False)  # Normal speed for place above
        state = "lower_to_place"
        timer = 0

    elif state == "lower_to_place":
        place_target = goal_position.copy()
        place_target[2] += 0.025
        smooth_move(place_target, slow_mode=True)  # Slow movement for precise placement
        state = "release"
        timer = 0

    elif state == "release":
        if timer > policy["release_delay"]:
            # Open PR2 gripper before releasing constraint
            open_gripper()
            
            # Step simulation to allow gripper to open
            for _ in range(30):
                p.stepSimulation()
                time.sleep(1/240)
            
            if constraint_id is not None:
                p.removeConstraint(constraint_id)
                constraint_id = None
            state = "observe"
            timer = 0
            stable_counter = 0

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

        print("\n========== LLM REFLECTION ==========")
        
        # Update GUI to show reflecting status
        gui_status.update_status("Reflecting", "LLM agent analyzing failure and planning next attempt")
        gui_status.display_status()
        log_robot_state(logger, "REFLECTING", "LLM agent analyzing failure and planning next attempt", retry_count + 1)

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
