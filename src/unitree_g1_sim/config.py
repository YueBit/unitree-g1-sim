"""
Unitree G1 Robot — Hardware Specification Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reference: Unitree G1 official technical documentation.
Joint limits, torque ratings, and battery specs used by the simulator.

:copyright: (c) 2025 ChenYue
:license: Apache 2.0
"""

# ── Joint definitions (29 DOF) ────────────────────────────────────
# fmt: off
G1_JOINTS = [
    # Left leg (6 DOF)
    {"name": "left_hip_pitch",   "id": 0,  "limits": (-2.87, 2.87), "max_velocity": 20.0, "max_torque": 88.0,  "group": "left_leg"},
    {"name": "left_hip_roll",    "id": 1,  "limits": (-0.52, 2.35), "max_velocity": 20.0, "max_torque": 88.0,  "group": "left_leg"},
    {"name": "left_hip_yaw",     "id": 2,  "limits": (-2.75, 2.75), "max_velocity": 20.0, "max_torque": 88.0,  "group": "left_leg"},
    {"name": "left_knee",        "id": 3,  "limits": (-0.26, 2.87), "max_velocity": 20.0, "max_torque": 139.0, "group": "left_leg"},
    {"name": "left_ankle_pitch", "id": 4,  "limits": (-0.90, 0.70), "max_velocity": 20.0, "max_torque": 50.0,  "group": "left_leg"},
    {"name": "left_ankle_roll",  "id": 5,  "limits": (-0.35, 0.35), "max_velocity": 20.0, "max_torque": 50.0,  "group": "left_leg"},
    # Right leg (6 DOF)
    {"name": "right_hip_pitch",  "id": 6,  "limits": (-2.87, 2.87), "max_velocity": 20.0, "max_torque": 88.0,  "group": "right_leg"},
    {"name": "right_hip_roll",   "id": 7,  "limits": (-2.35, 0.52), "max_velocity": 20.0, "max_torque": 88.0,  "group": "right_leg"},
    {"name": "right_hip_yaw",    "id": 8,  "limits": (-2.75, 2.75), "max_velocity": 20.0, "max_torque": 88.0,  "group": "right_leg"},
    {"name": "right_knee",       "id": 9,  "limits": (-0.26, 2.87), "max_velocity": 20.0, "max_torque": 139.0, "group": "right_leg"},
    {"name": "right_ankle_pitch","id": 10, "limits": (-0.90, 0.70), "max_velocity": 20.0, "max_torque": 50.0,  "group": "right_leg"},
    {"name": "right_ankle_roll", "id": 11, "limits": (-0.35, 0.35), "max_velocity": 20.0, "max_torque": 50.0,  "group": "right_leg"},
    # Waist (3 DOF)
    {"name": "waist_yaw",        "id": 12, "limits": (-2.62, 2.62), "max_velocity": 20.0, "max_torque": 88.0,  "group": "waist"},
    {"name": "waist_roll",       "id": 13, "limits": (-0.52, 0.52), "max_velocity": 20.0, "max_torque": 88.0,  "group": "waist"},
    {"name": "waist_pitch",      "id": 14, "limits": (-0.52, 0.52), "max_velocity": 20.0, "max_torque": 88.0,  "group": "waist"},
    # Left arm (7 DOF)
    {"name": "left_shoulder_pitch","id": 15,"limits": (-3.14, 3.14), "max_velocity": 20.0, "max_torque": 25.0,  "group": "left_arm"},
    {"name": "left_shoulder_roll", "id": 16,"limits": (-1.57, 3.14), "max_velocity": 20.0, "max_torque": 25.0,  "group": "left_arm"},
    {"name": "left_shoulder_yaw",  "id": 17,"limits": (-3.14, 3.14), "max_velocity": 20.0, "max_torque": 25.0,  "group": "left_arm"},
    {"name": "left_elbow",         "id": 18,"limits": (-1.57, 4.71), "max_velocity": 20.0, "max_torque": 25.0,  "group": "left_arm"},
    {"name": "left_wrist_roll",    "id": 19,"limits": (-3.14, 3.14), "max_velocity": 20.0, "max_torque": 5.0,   "group": "left_arm"},
    {"name": "left_wrist_pitch",   "id": 20,"limits": (-1.57, 1.57), "max_velocity": 20.0, "max_torque": 5.0,   "group": "left_arm"},
    {"name": "left_wrist_yaw",     "id": 21,"limits": (-1.57, 1.57), "max_velocity": 20.0, "max_torque": 5.0,   "group": "left_arm"},
    # Right arm (7 DOF)
    {"name": "right_shoulder_pitch","id": 22,"limits": (-3.14, 3.14),"max_velocity": 20.0, "max_torque": 25.0,  "group": "right_arm"},
    {"name": "right_shoulder_roll", "id": 23,"limits": (-3.14, 1.57),"max_velocity": 20.0, "max_torque": 25.0,  "group": "right_arm"},
    {"name": "right_shoulder_yaw",  "id": 24,"limits": (-3.14, 3.14),"max_velocity": 20.0, "max_torque": 25.0,  "group": "right_arm"},
    {"name": "right_elbow",         "id": 25,"limits": (-1.57, 4.71),"max_velocity": 20.0, "max_torque": 25.0,  "group": "right_arm"},
    {"name": "right_wrist_roll",    "id": 26,"limits": (-3.14, 3.14),"max_velocity": 20.0, "max_torque": 5.0,   "group": "right_arm"},
    {"name": "right_wrist_pitch",   "id": 27,"limits": (-1.57, 1.57),"max_velocity": 20.0, "max_torque": 5.0,   "group": "right_arm"},
    {"name": "right_wrist_yaw",     "id": 28,"limits": (-1.57, 1.57),"max_velocity": 20.0, "max_torque": 5.0,   "group": "right_arm"},
]
# fmt: on

# ── General specifications ────────────────────────────────────────

G1_SPECS = {
    "model": "Unitree G1",
    "dof": 29,
    "height_m": 1.27,
    "weight_kg": 35.0,
    "battery_voltage_nominal": 48.0,
    "battery_voltage_min": 42.0,
    "battery_voltage_max": 54.6,
    "battery_capacity_wh": 864.0,
    "max_joint_temp_celsius": 80.0,
    "warning_joint_temp_celsius": 65.0,
    "imu_gyro_range_dps": 2000.0,
    "imu_accel_range_g": 16.0,
}

# ── Lookup tables ──────────────────────────────────────────────────

JOINT_NAME_MAP: dict[str, dict] = {j["name"]: j for j in G1_JOINTS}
JOINT_GROUPS: list[str] = sorted({j["group"] for j in G1_JOINTS})
