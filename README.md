# unitree-g1-sim

> Lightweight pure-Python diagnostic simulator for the **Unitree G1** (29-DOF) humanoid robot.
> No MuJoCo. No Isaac Sim. No ROS. Just `pip install` and go.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://pypi.org/project/unitree-g1-sim/)

## Why?

Existing G1 simulators require heavy physics engines (MuJoCo, Isaac Sim).
This one is designed for **diagnostic testing** — CI/CD pipelines, hardware-less
development, and robot health monitoring systems. It generates realistic
joint/IMU/power telemetry across all 29 DOF, with configurable fault injection.

## Install

```bash
pip install unitree-g1-sim
```

The only dependency is `numpy`. No GPU required.

## Quick Start

```python
import asyncio
from unitree_g1_sim import G1Simulator, RobotState, FaultType

async def main():
    sim = G1Simulator(seed=42)   # seed for reproducibility

    await sim.connect()           # OFFLINE → STANDBY
    sim.set_state(RobotState.STANDING)

    # Read telemetry
    tele = sim.get_telemetry()
    print(f"Joints: {len(tele.joints)}")
    print(f"IMU roll: {tele.imu.roll:.5f} rad")
    print(f"Battery: {tele.power.soc:.1f}%")

    # Inject a fault
    sim.inject_fault(FaultType.JOINT_OVERTEMP)

    await sim.disconnect()

asyncio.run(main())
```

## Features

- **29 joints** with realistic kinematics and hardware limits
- **IMU simulation** (accel, gyro) with configurable noise models
- **Battery model** with Coulomb counting and voltage sag
- **Thermal model** with load-based heating and natural cooling
- **Fault injection**: joint overtemperature, position error, IMU drift, low battery, comm timeout
- **Seedable randomness** for deterministic, reproducible telemetry
- **5 built-in diagnostic tests** (TC-001 to TC-005: ROM, temp, IMU, stability, power)
- **Async-first** API for integration with asyncio applications

## Built-in Diagnostic Tests

```python
from unitree_g1_sim import G1Simulator
from unitree_g1_sim.builtin_tests import TEST_CATALOG

sim = G1Simulator(seed=42)
await sim.connect()

for tc_id, entry in TEST_CATALOG.items():
    result = await entry["fn"](sim)
    print(f"{result.status}  {result.test_name}: {result.summary}")
```

| ID | Name | Category |
|----|------|----------|
| TC-001 | Joint Range of Motion | Mechanics |
| TC-002 | Joint Temperature | Thermal |
| TC-003 | IMU Calibration | Sensors |
| TC-004 | Standing Stability | Control |
| TC-005 | Power Consumption | Power |

## API Reference

### `G1Simulator(seed: int | None = None)`

The main simulator class.

| Method | Description |
|--------|-------------|
| `await connect()` | OFFLINE → STANDBY |
| `await disconnect()` | Return to OFFLINE |
| `set_state(state)` | Set `RobotState` (STANDING, WALKING, TESTING, ...) |
| `get_telemetry()` | Generate one `RobotTelemetry` frame |
| `inject_fault(fault)` | Activate a `FaultType` |
| `clear_faults()` | Remove all faults |

### Data Models

- `RobotTelemetry` — timestamp, state, joints, imu, power, active_faults, uptime
- `JointTelemetry` — name, position, velocity, torque, temperature, error_code
- `IMUTelemetry` — roll, pitch, yaw, gyro_*, accel_*
- `PowerTelemetry` — voltage, current, soc (%), power_w

### Enums

- `RobotState` — OFFLINE, INITIALIZING, STANDBY, STANDING, WALKING, TESTING, FAULT, E_STOP
- `FaultType` — NONE, JOINT_OVERTEMP, JOINT_POSITION_ERROR, IMU_DRIFT, LOW_BATTERY, COMM_TIMEOUT

### Config

- `G1_JOINTS` — list of 29 joint definitions (name, id, limits, max_velocity, max_torque, group)
- `G1_SPECS` — dict of hardware specs (height, weight, battery, temp limits, etc.)

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Related

- [Unitree Robotics](https://github.com/unitreerobotics) — official G1 repos
- [RoboDiag](https://github.com/YueBit/RoboDiag) — AI diagnostic platform (uses unitree-g1-sim)
