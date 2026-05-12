import pybullet as p
import pybullet_data
import time
import os

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

p.loadURDF("plane.urdf")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(BASE_DIR, "urdf", "mycobot_320.urdf")

robot = p.loadURDF(
    URDF_PATH,
    useFixedBase=True,
    flags=p.URDF_USE_INERTIA_FROM_FILE
)

print("\n--- joint motion test (fixed) ---")

for j in range(p.getNumJoints(robot)):
    info = p.getJointInfo(robot, j)

    if info[2] != p.JOINT_REVOLUTE:
        continue

    name = info[1].decode()
    lower = info[8]
    upper = info[9]

    if lower >= upper:
        lower, upper = -1.0, 1.0

    print(f"\ntesting joint {j}: {name}")

    # move to lower
    for _ in range(240):
        p.setJointMotorControl2(
            robot, j,
            p.POSITION_CONTROL,
            targetPosition=lower,
            force=800
        )
        p.stepSimulation()
        time.sleep(1/240)

    time.sleep(0.5)

    # move to upper
    for _ in range(240):
        p.setJointMotorControl2(
            robot, j,
            p.POSITION_CONTROL,
            targetPosition=upper,
            force=800
        )
        p.stepSimulation()
        time.sleep(1/240)

    time.sleep(0.5)

    # HARD reset back to zero (key fix)
    p.resetJointState(robot, j, 0.0)
    p.setJointMotorControl2(
        robot, j,
        p.POSITION_CONTROL,
        targetPosition=0.0,
        force=800
    )

    time.sleep(1)

print("\n--- test complete ---")

while True:
    p.stepSimulation()
    time.sleep(1/240)
