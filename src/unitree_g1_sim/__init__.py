"""
unitree-g1-sim: Lightweight diagnostic simulator for Unitree G1 humanoid robot.
"""

from .config import G1_JOINTS, G1_SPECS, JOINT_GROUPS, JOINT_NAME_MAP
from .simulator import (
    FaultType,
    G1Simulator,
    IMUTelemetry,
    JointTelemetry,
    PowerTelemetry,
    RobotState,
    RobotTelemetry,
    create_simulator,
)
from .builtin_tests import (
    TEST_CATALOG,
    TestResult,
    TestStatus,
    TestStep,
)

__all__ = [
    "G1Simulator",
    "create_simulator",
    "RobotState",
    "FaultType",
    "JointTelemetry",
    "IMUTelemetry",
    "PowerTelemetry",
    "RobotTelemetry",
    "G1_JOINTS",
    "G1_SPECS",
    "JOINT_NAME_MAP",
    "JOINT_GROUPS",
    "TEST_CATALOG",
    "TestResult",
    "TestStatus",
    "TestStep",
]
__version__ = "0.1.0"
