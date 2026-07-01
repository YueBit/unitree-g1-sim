"""
Unitree G1 Humanoid Robot — Diagnostic Simulator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A lightweight, pure-Python simulator that generates realistic
telemetry data for the Unitree G1 (29-DOF) humanoid robot.
Supports configurable fault injection for testing diagnostic algorithms.

Usage::

    from unitree_g1_sim import G1Simulator

    sim = G1Simulator(seed=42)
    await sim.connect()
    sim.set_state(RobotState.STANDING)
    tele = sim.get_telemetry()
    print(tele.joints[3].temperature)

:copyright: (c) 2025 ChenYue
:license: Apache 2.0, see LICENSE for details.
"""

import asyncio
import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import numpy as np

from .config import G1_JOINTS, G1_SPECS

# ═══════════════════════════════════════════════════════════════════
#  Physics constants (tuned for G1's 1.27m / 35kg form factor)
# ═══════════════════════════════════════════════════════════════════

# ── Joint motion ──
_WALKING_FREQ_HZ = 1.2          # G1 walking cadence
_WALKING_LEG_AMP = 0.4          # leg joint amplitude (rad)
_WALKING_NONLEG_AMP = 0.15      # non-leg joint amplitude
_STANDING_JITTER_NOISE = 0.002  # standing balance noise (σ, rad)
_JOINT_NOISE_SIGMA = 0.002      # general position noise
_TORQUE_NOISE_SIGMA = 0.5       # torque noise (Nm)
_GRAVITY_NOISE_KNEE = 1.0       # knee torque noise
_GRAVITY_NOISE_HIP = 0.5        # hip torque noise
_GRAVITY_NOISE_ANKLE = 0.3      # ankle torque noise
_GRAVITY_NOISE_OTHER = 0.2      # other joint torque noise

# ── Gravity compensation ──
_GRAVITY = 9.81
_KNEE_GRAVITY_RATIO = 0.15      # fraction of body weight on knee
_HIP_PITCH_GRAVITY_RATIO = 0.08
_ANKLE_GRAVITY_RATIO = 0.05

# ── Thermal model ──
_AMBIENT_TEMP_C = 22.0          # ambient temperature floor
_INITIAL_TEMP_C = 28.0          # starting joint temperature
_INITIAL_TEMP_SPREAD = 3.0      # ± spread for initial temps
_TEMP_HEATING_RATE = 0.005      # °C per tick per unit load
_TEMP_COOLING_RATE = 0.001      # °C natural cooling per tick
_OVERTEMP_RAMP = 0.5            # °C per tick when fault injected
_OVERTEMP_CEILING = 92.0        # hard ceiling for injected fault
_TEMP_SENSOR_NOISE = 0.15       # sensor quantization noise (σ, °C)
_TEMP_THERMAL_LAG = 0.02        # thermal lag coefficient (smaller = more lag)

# ── Battery model ──
_INITIAL_SOC_PCT = 85.0         # starting SOC (%)
_INITIAL_SOC_SPREAD = 10.0      # ± spread
_WALKING_CURRENT_A = 18.0       # nominal current draw
_WALKING_CURRENT_NOISE = 1.5
_STANDING_CURRENT_A = 8.0
_STANDING_CURRENT_NOISE = 0.5
_IDLE_CURRENT_A = 2.5
_IDLE_CURRENT_NOISE = 0.2
_TELEMETRY_DT_S = 0.1           # assumed call interval for SOC decay

# ── IMU ──
_STANDING_IMU_NOISE = 0.008     # roll/pitch noise (σ, rad)
_WALKING_IMU_NOISE = 0.01       # walking noise
_IDLE_IMU_NOISE = 0.003         # idle noise
_IMU_GYRO_NOISE = 0.01          # gyro noise σ (rad/s)
_IMU_GYRO_Z_NOISE = 0.005
_IMU_ACCEL_NOISE = 0.05         # accel noise σ (m/s²)
_IMU_ACCEL_Z_NOISE = 0.03       # z-axis accel noise
_IMU_DRIFT_AMP = 0.15           # injected drift amplitude

# ── Walking gait ──
_WALKING_AMP_STANDING = 0.05    # standing sway (rad)
_WALKING_PHASE_STANDING = 0.3   # standing sway frequency multiplier
_IDLE_NOISE_SCALE = 0.5         # idle position noise attenuator

# ── IMU walking ──
_IMU_WALKING_ROLL_AMP = 0.06
_IMU_WALKING_PITCH_AMP = 0.04
_IMU_WALKING_PITCH_PHASE = 0.5  # pitch offset from roll
_IMU_YAW_DRIFT_RATE = 0.02      # yaw drift rad/s while walking

# ── Joint fault injection targets ──
_FAULT_POSITION_OFFSET_MIN = 0.3
_FAULT_POSITION_OFFSET_MAX = 0.6
_FAULT_ERROR_CODE = 0x04           # 关节位置错误
_OVERTEMP_JOINT_IDS = (3, 9)       # left/right knees (high-load joints)
_POSITION_ERROR_JOINT_IDS = (3, 9) # same targets for position fault demo

# ── Comm timeout ──
_COMM_TIMEOUT_DROP_PROB = 0.3      # probability of dropping a frame during timeout
_COMM_TIMEOUT_ZERO_PROB = 0.5      # probability of zeroing joint data

# ═══════════════════════════════════════════════════════════════════
#  Enums
# ═══════════════════════════════════════════════════════════════════


class RobotState(str, Enum):
    """Robot operational state."""

    OFFLINE = "offline"
    INITIALIZING = "initializing"
    STANDBY = "standby"
    STANDING = "standing"
    WALKING = "walking"
    TESTING = "testing"
    FAULT = "fault"
    E_STOP = "e_stop"


class FaultType(str, Enum):
    """Supported fault injection types."""

    NONE = "none"
    JOINT_OVERTEMP = "joint_overtemp"
    JOINT_POSITION_ERROR = "joint_position_error"
    IMU_DRIFT = "imu_drift"
    LOW_BATTERY = "low_battery"
    COMM_TIMEOUT = "comm_timeout"

# ═══════════════════════════════════════════════════════════════════
#  Telemetry dataclasses
# ═══════════════════════════════════════════════════════════════════


@dataclass
class JointTelemetry:
    """Per-joint telemetry snapshot."""

    name: str
    position: float       # rad
    velocity: float       # rad/s
    torque: float         # Nm
    temperature: float    # °C
    error_code: int = 0   # 0 = normal


@dataclass
class IMUTelemetry:
    """IMU (Inertial Measurement Unit) telemetry."""

    roll: float           # rad
    pitch: float          # rad
    yaw: float            # rad
    gyro_x: float         # rad/s
    gyro_y: float         # rad/s
    gyro_z: float         # rad/s
    accel_x: float        # m/s²
    accel_y: float        # m/s²
    accel_z: float        # m/s²


@dataclass
class PowerTelemetry:
    """Battery and power system telemetry."""

    voltage: float        # V
    current: float        # A
    soc: float            # 0-100 %
    power_w: float        # W


@dataclass
class RobotTelemetry:
    """Complete robot telemetry frame."""

    timestamp: float
    state: RobotState
    joints: list[JointTelemetry]
    imu: IMUTelemetry
    power: PowerTelemetry
    active_faults: list[str] = field(default_factory=list)
    uptime_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-serializable dict (for recording/export)."""
        return {
            "timestamp": self.timestamp,
            "state": self.state.value,
            "joints": [
                {
                    "name": j.name, "position": j.position,
                    "velocity": j.velocity, "torque": j.torque,
                    "temperature": j.temperature, "error_code": j.error_code,
                }
                for j in self.joints
            ],
            "imu": {
                "roll": self.imu.roll, "pitch": self.imu.pitch,
                "yaw": self.imu.yaw,
                "gyro_x": self.imu.gyro_x, "gyro_y": self.imu.gyro_y,
                "gyro_z": self.imu.gyro_z,
                "accel_x": self.imu.accel_x, "accel_y": self.imu.accel_y,
                "accel_z": self.imu.accel_z,
            },
            "power": {
                "voltage": self.power.voltage, "current": self.power.current,
                "soc": self.power.soc, "power_w": self.power.power_w,
            },
            "active_faults": self.active_faults,
            "uptime_s": self.uptime_s,
        }

# ═══════════════════════════════════════════════════════════════════
#  Simulator
# ═══════════════════════════════════════════════════════════════════


class G1Simulator:
    """G1 robot diagnostic simulator.

    Generates realistic telemetry data across all 29 joints,
    including IMU, power, and temperature.  Supports injecting
    faults to test diagnostic algorithms.

    Parameters
    ----------
    seed:
        Random seed for reproducible telemetry.  Pass an ``int``
        to get deterministic output; omit for random behaviour.
    """

    def __init__(self, seed: int | None = None):
        self.state = RobotState.OFFLINE
        self._start_tick: int = 0       # tick at which connect() was called
        self._rng = random.Random(seed)
        self._np_rng = np.random.RandomState(seed)
        self._joint_states: dict[str, dict] = {}
        self._battery_soc = _INITIAL_SOC_PCT + self._rng.uniform(
            -_INITIAL_SOC_SPREAD, _INITIAL_SOC_SPREAD
        )
        self._inject_faults: list[FaultType] = []
        self._running = False
        self._telemetry_callbacks: list[Callable[[RobotTelemetry], None]] = []
        self._tick = 0
        self._lock = asyncio.Lock()
        # Temperature sensor: store "true" temp and "reported" temp with thermal lag
        self._true_temps: dict[str, float] = {}
        self._reported_temps: dict[str, float] = {}
        # Recording buffer
        self._recording: list[dict[str, Any]] | None = None
        self._init_joint_states()

    # ── internal init ──────────────────────────────────────────────

    def _init_joint_states(self) -> None:
        """Populate per-joint state dictionaries from G1_JOINTS config."""
        for j in G1_JOINTS:
            name = j["name"]
            init_temp = _INITIAL_TEMP_C + self._rng.uniform(
                -_INITIAL_TEMP_SPREAD, _INITIAL_TEMP_SPREAD
            )
            self._joint_states[name] = {
                "position": 0.0,
                "target": 0.0,
                "velocity": 0.0,
                "torque": 0.0,
                "temperature": init_temp,
                "error_code": 0,
                "limits": j["limits"],
                "max_velocity": j["max_velocity"],
                "max_torque": j["max_torque"],
                "group": j["group"],
            }
            self._true_temps[name] = init_temp
            self._reported_temps[name] = init_temp

    # ── public API ─────────────────────────────────────────────────

    async def connect(self) -> None:
        """Transition from OFFLINE → INITIALIZING → STANDBY."""
        async with self._lock:
            self.state = RobotState.INITIALIZING
            self._start_tick = 0
            self._battery_soc = _INITIAL_SOC_PCT + self._rng.uniform(
                -_INITIAL_SOC_SPREAD, _INITIAL_SOC_SPREAD
            )
            self._tick = 0      # reset tick counter on (re)connect
            await asyncio.sleep(0.5)
            self.state = RobotState.STANDBY
            self._running = True

    async def disconnect(self) -> None:
        """Shut down and return to OFFLINE."""
        async with self._lock:
            self._running = False
            self.state = RobotState.OFFLINE

    def set_state(self, state: RobotState) -> None:
        """Set robot operational state."""
        self.state = state

    def inject_fault(self, fault: FaultType) -> None:
        """Activate a fault type (idempotent)."""
        if fault not in self._inject_faults:
            self._inject_faults.append(fault)

    def clear_faults(self) -> None:
        """Remove all injected faults and reset joint error codes."""
        self._inject_faults.clear()
        for name, j in self._joint_states.items():
            j["error_code"] = 0
            if j["temperature"] > G1_SPECS["max_joint_temp_celsius"]:
                j["temperature"] = 35.0
                self._true_temps[name] = 35.0
                self._reported_temps[name] = 35.0

    # ── callbacks & recording ─────────────────────────────────────

    def on_telemetry(self, callback: Callable[[RobotTelemetry], None]) -> None:
        """Register a callback to be invoked on every telemetry frame.

        Useful for WebSocket streaming, logging, or real-time dashboards.
        Callbacks are called synchronously inside ``get_telemetry()``.

        Example::

            sim.on_telemetry(lambda tele: websocket.send(tele.to_dict()))
        """
        self._telemetry_callbacks.append(callback)

    def remove_callback(self, callback: Callable[[RobotTelemetry], None]) -> None:
        """Remove a previously registered callback."""
        if callback in self._telemetry_callbacks:
            self._telemetry_callbacks.remove(callback)

    def _notify_callbacks(self, tele: RobotTelemetry) -> None:
        """Invoke all registered telemetry callbacks."""
        for cb in self._telemetry_callbacks:
            try:
                cb(tele)
            except Exception:
                pass  # Don't let one broken callback break the pipeline

    def start_recording(self) -> None:
        """Begin recording telemetry frames to an in-memory buffer.

        Call ``stop_recording()`` to retrieve the recorded frames.
        """
        self._recording = []

    def stop_recording(self) -> list[dict[str, Any]]:
        """Stop recording and return all captured frames as dicts.

        Returns an empty list if recording was not active.
        """
        frames = self._recording or []
        self._recording = None
        return frames

    def is_recording(self) -> bool:
        """Return True if recording is currently active."""
        return self._recording is not None

    def get_telemetry(self) -> RobotTelemetry:
        """Generate one telemetry frame. Fires registered callbacks."""
        self._tick += 1
        t = time.time()
        uptime = (self._tick - self._start_tick) * _TELEMETRY_DT_S

        # COMM_TIMEOUT: randomly drop frames or corrupt data
        if FaultType.COMM_TIMEOUT in self._inject_faults:
            if self._rng.random() < _COMM_TIMEOUT_DROP_PROB:
                tele = RobotTelemetry(
                    timestamp=t, state=self.state,
                    joints=[], imu=IMUTelemetry(0,0,0,0,0,0,0,0,0),
                    power=PowerTelemetry(0,0,0,0),
                    active_faults=["COMM_TIMEOUT:通信中断-丢帧"],
                    uptime_s=uptime,
                )
                self._notify_callbacks(tele)
                return tele

        joints = self._compute_joints(uptime)
        imu = self._compute_imu(uptime)
        power = self._compute_power(uptime)
        faults = self._get_active_faults(joints, imu, power)

        if FaultType.COMM_TIMEOUT in self._inject_faults and self._rng.random() < _COMM_TIMEOUT_ZERO_PROB:
            for j in joints:
                j.position = 0.0
                j.velocity = 0.0
                j.torque = 0.0

        if faults:
            self.state = RobotState.FAULT

        tele = RobotTelemetry(
            timestamp=t,
            state=self.state,
            joints=joints,
            imu=imu,
            power=power,
            active_faults=faults,
            uptime_s=uptime,
        )

        self._notify_callbacks(tele)

        if self._recording is not None:
            self._recording.append(tele.to_dict())

        return tele

    # ── joint kinematics ───────────────────────────────────────────

    def _compute_joints(self, uptime: float) -> list[JointTelemetry]:
        """Compute joint positions/velocities/torques/temps for one frame."""
        result: list[JointTelemetry] = []
        for j_config in G1_JOINTS:
            s = self._joint_states[j_config["name"]]
            group = j_config["group"]
            name = j_config["name"]
            jid = j_config["id"]

            noise = self._np_rng.normal(0, _JOINT_NOISE_SIGMA)

            if self.state == RobotState.STANDING:
                pos = (
                    math.sin(uptime * _WALKING_PHASE_STANDING + jid * 0.2)
                    * _WALKING_AMP_STANDING
                    + noise
                )
                vel = (
                    math.cos(uptime * _WALKING_PHASE_STANDING + jid * 0.2)
                    * _WALKING_AMP_STANDING
                )
                torque = self._gravity_torque(name) + self._np_rng.normal(0, _TORQUE_NOISE_SIGMA)
            elif self.state in (RobotState.WALKING, RobotState.TESTING):
                freq = _WALKING_FREQ_HZ if group in ("left_leg", "right_leg") else 0.5
                amp = (
                    _WALKING_LEG_AMP
                    if "knee" in name or "hip" in name or "ankle" in name
                    else _WALKING_NONLEG_AMP
                )
                pos = (
                    math.sin(uptime * freq * 2 * math.pi + jid * 0.7) * amp
                    + noise
                )
                vel = (
                    math.cos(uptime * freq * 2 * math.pi + jid * 0.7)
                    * amp
                    * freq
                    * 2
                    * math.pi
                )
                torque = self._gravity_torque(name) * (
                    1 + 0.3 * math.sin(uptime * freq)
                )
            else:
                pos = noise * _IDLE_NOISE_SCALE
                vel = 0.0
                torque = self._gravity_torque(name) * 0.1

            # ── clamp to hardware limits ──
            lo, hi = j_config["limits"]
            pos = max(lo, min(hi, pos))
            vel = max(-j_config["max_velocity"], min(j_config["max_velocity"], vel))
            torque = max(
                -j_config["max_torque"], min(j_config["max_torque"], torque)
            )

            # ── thermal model (with sensor noise + thermal lag) ──
            load_factor = abs(torque) / max(j_config["max_torque"], 1)
            # True physical temperature
            true_temp: float = self._true_temps.get(name, float(s["temperature"]))
            true_temp += load_factor * _TEMP_HEATING_RATE - _TEMP_COOLING_RATE
            true_temp = max(_AMBIENT_TEMP_C, true_temp)
            self._true_temps[name] = true_temp
            # Reported temperature: lagged true temp + sensor noise
            reported: float = self._reported_temps.get(name, true_temp)
            reported += (true_temp - reported) * _TEMP_THERMAL_LAG
            reported += self._np_rng.normal(0, _TEMP_SENSOR_NOISE)
            reported = max(_AMBIENT_TEMP_C, reported)
            self._reported_temps[name] = reported
            s["temperature"] = reported

            # ── fault injection ──
            error_code = 0
            if (
                FaultType.JOINT_OVERTEMP in self._inject_faults
                and jid in _OVERTEMP_JOINT_IDS
            ):
                self._true_temps[name] = min(
                    true_temp + _OVERTEMP_RAMP, _OVERTEMP_CEILING
                )
                self._reported_temps[name] = self._true_temps[name]
                s["temperature"] = self._true_temps[name]
            if (
                FaultType.JOINT_POSITION_ERROR in self._inject_faults
                and jid in _POSITION_ERROR_JOINT_IDS
            ):
                pos += self._rng.uniform(
                    _FAULT_POSITION_OFFSET_MIN, _FAULT_POSITION_OFFSET_MAX
                )
                error_code = _FAULT_ERROR_CODE

            s["position"] = pos
            s["velocity"] = vel
            s["torque"] = torque

            result.append(
                JointTelemetry(
                    name=name,
                    position=round(pos, 4),
                    velocity=round(vel, 4),
                    torque=round(torque, 3),
                    temperature=round(s["temperature"], 1),
                    error_code=error_code,
                )
            )
        return result

    def _gravity_torque(self, joint_name: str) -> float:
        """Approximate gravity compensation torque for a joint."""
        weight = G1_SPECS["weight_kg"] * _GRAVITY
        if "knee" in joint_name:
            return weight * _KNEE_GRAVITY_RATIO + self._np_rng.normal(
                0, _GRAVITY_NOISE_KNEE
            )
        if "hip_pitch" in joint_name:
            return weight * _HIP_PITCH_GRAVITY_RATIO + self._np_rng.normal(
                0, _GRAVITY_NOISE_HIP
            )
        if "ankle" in joint_name:
            return weight * _ANKLE_GRAVITY_RATIO + self._np_rng.normal(
                0, _GRAVITY_NOISE_ANKLE
            )
        return self._np_rng.normal(0, _GRAVITY_NOISE_OTHER)

    # ── IMU ────────────────────────────────────────────────────────

    def _compute_imu(self, uptime: float) -> IMUTelemetry:
        """Generate IMU readings based on current state."""
        if self.state == RobotState.STANDING:
            roll = self._np_rng.normal(0, _STANDING_IMU_NOISE)
            pitch = self._np_rng.normal(0, _STANDING_IMU_NOISE)
        elif self.state in (RobotState.WALKING, RobotState.TESTING):
            roll = math.sin(uptime * _WALKING_FREQ_HZ * 2 * math.pi) * _IMU_WALKING_ROLL_AMP + self._np_rng.normal(0, _WALKING_IMU_NOISE)
            pitch = (
                math.sin(
                    uptime * _WALKING_FREQ_HZ * 2 * math.pi + _IMU_WALKING_PITCH_PHASE
                )
                * _IMU_WALKING_PITCH_AMP
                + self._np_rng.normal(0, _WALKING_IMU_NOISE)
            )
        else:
            roll = self._np_rng.normal(0, _IDLE_IMU_NOISE)
            pitch = self._np_rng.normal(0, _IDLE_IMU_NOISE)

        yaw = (
            uptime * (_IMU_YAW_DRIFT_RATE if self.state == RobotState.WALKING else 0.0)
            + self._np_rng.normal(0, 0.002)
        )

        if FaultType.IMU_DRIFT in self._inject_faults:
            roll += math.sin(uptime * 0.1) * _IMU_DRIFT_AMP
            pitch += math.cos(uptime * 0.1) * _IMU_DRIFT_AMP * 0.8

        return IMUTelemetry(
            roll=round(roll, 5),
            pitch=round(pitch, 5),
            yaw=round(yaw % (2 * math.pi), 5),
            gyro_x=round(self._np_rng.normal(0, _IMU_GYRO_NOISE), 5),
            gyro_y=round(self._np_rng.normal(0, _IMU_GYRO_NOISE), 5),
            gyro_z=round(self._np_rng.normal(0, _IMU_GYRO_Z_NOISE), 5),
            accel_x=round(self._np_rng.normal(0, _IMU_ACCEL_NOISE), 4),
            accel_y=round(self._np_rng.normal(0, _IMU_ACCEL_NOISE), 4),
            accel_z=round(_GRAVITY + self._np_rng.normal(0, _IMU_ACCEL_Z_NOISE), 4),
        )

    # ── power ──────────────────────────────────────────────────────

    def _compute_power(self, uptime: float) -> PowerTelemetry:  # noqa: ARG002
        """Estimate battery drain and voltage sag."""
        if self.state in (RobotState.WALKING, RobotState.TESTING):
            current = _WALKING_CURRENT_A + self._np_rng.normal(
                0, _WALKING_CURRENT_NOISE
            )
        elif self.state == RobotState.STANDING:
            current = _STANDING_CURRENT_A + self._np_rng.normal(
                0, _STANDING_CURRENT_NOISE
            )
        else:
            current = _IDLE_CURRENT_A + self._np_rng.normal(
                0, _IDLE_CURRENT_NOISE
            )

        # Coulomb counting: SOC drops proportionally to current draw
        self._battery_soc -= (
            current
            * (_TELEMETRY_DT_S / 3600)
            / G1_SPECS["battery_capacity_wh"]
            * G1_SPECS["battery_voltage_nominal"]
            * 100
        )
        self._battery_soc = max(0.0, min(100.0, self._battery_soc))

        voltage = G1_SPECS["battery_voltage_min"] + (
            G1_SPECS["battery_voltage_max"] - G1_SPECS["battery_voltage_min"]
        ) * (self._battery_soc / 100)

        if FaultType.LOW_BATTERY in self._inject_faults:
            self._battery_soc = max(3.0, self._battery_soc - 0.5)
            voltage = 42.5

        return PowerTelemetry(
            voltage=round(voltage, 2),
            current=round(abs(current), 2),
            soc=round(self._battery_soc, 1),
            power_w=round(abs(current) * voltage, 1),
        )

    # ── fault aggregation ──────────────────────────────────────────

    def _get_active_faults(
        self,
        joints: list[JointTelemetry],
        imu: IMUTelemetry,
        power: PowerTelemetry,
    ) -> list[str]:
        """Aggregate all active faults from joints, IMU, and power."""
        faults: list[str] = []

        # joint faults
        for j in joints:
            if j.temperature > G1_SPECS["max_joint_temp_celsius"]:
                faults.append(f"JOINT_OVERTEMP:{j.name}:{j.temperature:.1f}°C")
            if j.error_code != 0:
                faults.append(f"JOINT_ERROR:{j.name}:0x{j.error_code:02X}")

        # IMU faults
        if FaultType.IMU_DRIFT in self._inject_faults:
            faults.append("IMU_DRIFT:姿态估计偏差")

        # power faults
        if FaultType.LOW_BATTERY in self._inject_faults:
            faults.append(f"LOW_BATTERY:SOC={power.soc:.1f}%")

        if FaultType.COMM_TIMEOUT in self._inject_faults:
            faults.append("COMM_TIMEOUT:通信超时")

        return faults


# ═══════════════════════════════════════════════════════════════════
#  Convenience factory
# ═══════════════════════════════════════════════════════════════════

def create_simulator(seed: int | None = None) -> G1Simulator:
    """Create a new G1Simulator instance.

    Parameters
    ----------
    seed:
        Random seed for reproducibility.
    """
    return G1Simulator(seed=seed)
