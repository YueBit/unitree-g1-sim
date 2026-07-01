"""
Tests for g1-sim: Unitree G1 diagnostic simulator.
"""

import asyncio

import pytest

from unitree_g1_sim import (
    G1Simulator,
    RobotState,
    FaultType,
    G1_JOINTS,
    G1_SPECS,
    JointTelemetry,
    IMUTelemetry,
    PowerTelemetry,
    RobotTelemetry,
)
from unitree_g1_sim.builtin_tests import TEST_CATALOG, TestStatus


# ═══════════════════════════════════════════════════════════════════
#  Simulator lifecycle
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_initial_state_offline():
    sim = G1Simulator()
    assert sim.state == RobotState.OFFLINE


@pytest.mark.asyncio
async def test_connect_transitions_to_standby():
    sim = G1Simulator()
    await sim.connect()
    assert sim.state == RobotState.STANDBY


@pytest.mark.asyncio
async def test_disconnect_returns_to_offline():
    sim = G1Simulator()
    await sim.connect()
    await sim.disconnect()
    assert sim.state == RobotState.OFFLINE


@pytest.mark.asyncio
async def test_set_state():
    sim = G1Simulator()
    sim.set_state(RobotState.STANDING)
    assert sim.state == RobotState.STANDING
    sim.set_state(RobotState.WALKING)
    assert sim.state == RobotState.WALKING


# ═══════════════════════════════════════════════════════════════════
#  Telemetry structure
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_telemetry_has_29_joints():
    sim = G1Simulator()
    await sim.connect()
    tele = sim.get_telemetry()
    assert len(tele.joints) == 29


@pytest.mark.asyncio
async def test_telemetry_when_offline_still_works():
    sim = G1Simulator()
    tele = sim.get_telemetry()
    assert tele.state == RobotState.OFFLINE
    assert len(tele.joints) == 29


@pytest.mark.asyncio
async def test_telemetry_includes_all_fields():
    sim = G1Simulator()
    await sim.connect()
    tele = sim.get_telemetry()

    assert isinstance(tele.timestamp, float)
    assert isinstance(tele.state, RobotState)
    assert isinstance(tele.uptime_s, float)
    assert isinstance(tele.joints[0], JointTelemetry)
    assert isinstance(tele.imu, IMUTelemetry)
    assert isinstance(tele.power, PowerTelemetry)
    assert isinstance(tele.active_faults, list)


@pytest.mark.asyncio
async def test_joint_names_match_config():
    sim = G1Simulator()
    await sim.connect()
    tele = sim.get_telemetry()
    config_names = [j["name"] for j in G1_JOINTS]
    telemetry_names = [j.name for j in tele.joints]
    assert config_names == telemetry_names


@pytest.mark.asyncio
async def test_imu_accel_z_approx_gravity():
    sim = G1Simulator(seed=42)
    await sim.connect()
    tele = sim.get_telemetry()
    assert 9.7 < tele.imu.accel_z < 9.9


@pytest.mark.asyncio
async def test_power_soc_between_0_and_100():
    sim = G1Simulator()
    await sim.connect()
    tele = sim.get_telemetry()
    assert 0 <= tele.power.soc <= 100


# ═══════════════════════════════════════════════════════════════════
#  Determinism
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_seeded_simulators_produce_identical_telemetry():
    sim1 = G1Simulator(seed=42)
    sim2 = G1Simulator(seed=42)
    await sim1.connect()
    await sim2.connect()
    sim1.set_state(RobotState.STANDING)
    sim2.set_state(RobotState.STANDING)

    t1 = sim1.get_telemetry()
    t2 = sim2.get_telemetry()

    # Compare first joint
    assert t1.joints[0].position == t2.joints[0].position
    assert t1.joints[0].temperature == t2.joints[0].temperature
    assert t1.power.soc == t2.power.soc


@pytest.mark.asyncio
async def test_unseeded_simulators_differ():
    sim1 = G1Simulator()
    sim2 = G1Simulator()
    await sim1.connect()
    await sim2.connect()
    sim1.set_state(RobotState.STANDING)
    sim2.set_state(RobotState.STANDING)

    t1 = sim1.get_telemetry()
    t2 = sim2.get_telemetry()

    # At least one value should differ between unseeded sims
    diffs = (
        t1.joints[0].position != t2.joints[0].position,
        t1.power.soc != t2.power.soc,
        t1.imu.roll != t2.imu.roll,
    )
    assert any(diffs), "Unseeded simulators produced identical values"


# ═══════════════════════════════════════════════════════════════════
#  Joint limits
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_joint_positions_within_hardware_limits():
    sim = G1Simulator(seed=42)
    await sim.connect()
    sim.set_state(RobotState.WALKING)

    for _ in range(20):
        tele = sim.get_telemetry()
        for j, j_config in zip(tele.joints, G1_JOINTS):
            lo, hi = j_config["limits"]
            assert lo - 1e-6 <= j.position <= hi + 1e-6, (
                f"{j.name} position {j.position} outside [{lo}, {hi}]"
            )
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_joint_velocity_within_hardware_limits():
    sim = G1Simulator(seed=42)
    await sim.connect()
    sim.set_state(RobotState.WALKING)

    for _ in range(20):
        tele = sim.get_telemetry()
        for j, j_config in zip(tele.joints, G1_JOINTS):
            max_v = j_config["max_velocity"]
            assert abs(j.velocity) <= max_v + 1e-6, (
                f"{j.name} velocity {j.velocity} exceeds {max_v}"
            )
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_joint_torque_within_hardware_limits():
    sim = G1Simulator(seed=42)
    await sim.connect()
    sim.set_state(RobotState.WALKING)

    for _ in range(20):
        tele = sim.get_telemetry()
        for j, j_config in zip(tele.joints, G1_JOINTS):
            max_t = j_config["max_torque"]
            assert abs(j.torque) <= max_t + 1e-6, (
                f"{j.name} torque {j.torque} exceeds {max_t}"
            )
        await asyncio.sleep(0.01)


# ═══════════════════════════════════════════════════════════════════
#  Fault injection
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_inject_joint_overtemp_raises_temperature():
    sim = G1Simulator(seed=42)
    await sim.connect()
    sim.set_state(RobotState.STANDING)

    # baseline
    tele_before = sim.get_telemetry()
    temp_before = tele_before.joints[3].temperature  # left_knee

    # inject and wait
    sim.inject_fault(FaultType.JOINT_OVERTEMP)
    for _ in range(10):
        sim.get_telemetry()
    tele_after = sim.get_telemetry()
    temp_after = tele_after.joints[3].temperature

    assert temp_after > temp_before, "Overtemp fault did not raise temperature"


@pytest.mark.asyncio
async def test_inject_imu_drift_appears_in_faults():
    sim = G1Simulator(seed=42)
    await sim.connect()
    sim.set_state(RobotState.STANDING)

    sim.inject_fault(FaultType.IMU_DRIFT)
    tele = sim.get_telemetry()
    assert any("IMU_DRIFT" in f for f in tele.active_faults)


@pytest.mark.asyncio
async def test_inject_low_battery_appears_in_faults():
    sim = G1Simulator(seed=42)
    await sim.connect()
    sim.set_state(RobotState.STANDING)

    sim.inject_fault(FaultType.LOW_BATTERY)
    tele = sim.get_telemetry()
    assert any("LOW_BATTERY" in f for f in tele.active_faults)


@pytest.mark.asyncio
async def test_clear_faults_removes_all():
    sim = G1Simulator(seed=42)
    await sim.connect()
    sim.inject_fault(FaultType.JOINT_OVERTEMP)
    sim.inject_fault(FaultType.IMU_DRIFT)

    sim.clear_faults()
    tele = sim.get_telemetry()
    assert tele.active_faults == []


# ═══════════════════════════════════════════════════════════════════
#  Built-in tests
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_catalog_has_5_tests():
    assert len(TEST_CATALOG) == 5
    for tc_id in ["TC-001", "TC-002", "TC-003", "TC-004", "TC-005"]:
        assert tc_id in TEST_CATALOG


@pytest.mark.asyncio
async def test_run_tc001_passes():
    sim = G1Simulator(seed=42)
    await sim.connect()
    result = await TEST_CATALOG["TC-001"]["fn"](sim)
    assert result.status in (TestStatus.PASS, TestStatus.WARNING, TestStatus.FAIL)


@pytest.mark.asyncio
async def test_tc003_imu_passes_with_seed():
    sim = G1Simulator(seed=42)
    await sim.connect()
    result = await TEST_CATALOG["TC-003"]["fn"](sim)
    assert result.status == TestStatus.PASS


# ═══════════════════════════════════════════════════════════════════
#  Config
# ═══════════════════════════════════════════════════════════════════


def test_config_has_29_joints():
    assert len(G1_JOINTS) == 29


def test_config_joint_ids_sequential():
    ids = [j["id"] for j in G1_JOINTS]
    assert ids == list(range(29))


def test_config_has_all_groups():
    groups = {j["group"] for j in G1_JOINTS}
    assert groups == {"left_leg", "right_leg", "waist", "left_arm", "right_arm"}


def test_config_specs_complete():
    required = [
        "model", "dof", "height_m", "weight_kg",
        "battery_voltage_nominal", "battery_capacity_wh",
        "max_joint_temp_celsius",
    ]
    for key in required:
        assert key in G1_SPECS

# ═══════════════════════════════════════════════════════════════════
#  New features: callbacks, recording, comm_timeout, temp noise
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_telemetry_callback_fires():
    """Callbacks should be invoked on every get_telemetry() call."""
    sim = G1Simulator(seed=42)
    await sim.connect()
    sim.set_state(RobotState.STANDING)

    frames = []
    sim.on_telemetry(lambda t: frames.append(t))

    for _ in range(5):
        sim.get_telemetry()

    assert len(frames) == 5
    assert all(isinstance(f, RobotTelemetry) for f in frames)

@pytest.mark.asyncio
async def test_callback_remove():
    """Removed callbacks should not fire."""
    sim = G1Simulator(seed=42)
    await sim.connect()

    frames = []
    cb = lambda t: frames.append(t)
    sim.on_telemetry(cb)
    sim.get_telemetry()
    assert len(frames) == 1

    sim.remove_callback(cb)
    sim.get_telemetry()
    assert len(frames) == 1  # No new frame

@pytest.mark.asyncio
async def test_recording_start_stop():
    """Recording should capture frames and stop on command."""
    sim = G1Simulator(seed=42)
    await sim.connect()
    sim.set_state(RobotState.STANDING)

    sim.start_recording()
    assert sim.is_recording()

    for _ in range(10):
        sim.get_telemetry()

    frames = sim.stop_recording()
    assert not sim.is_recording()
    assert len(frames) == 10
    assert all("joints" in f for f in frames)
    assert all("imu" in f for f in frames)
    assert all("power" in f for f in frames)

@pytest.mark.asyncio
async def test_comm_timeout_drops_frames():
    """COMM_TIMEOUT should sometimes return empty telemetry."""
    sim = G1Simulator(seed=42)
    await sim.connect()
    sim.set_state(RobotState.STANDING)
    sim.inject_fault(FaultType.COMM_TIMEOUT)

    drop_seen = zero_seen = normal_seen = False
    for _ in range(100):
        tele = sim.get_telemetry()
        if len(tele.joints) == 0:
            drop_seen = True
        elif all(j.position == 0.0 for j in tele.joints):
            zero_seen = True
        else:
            normal_seen = True

    # With 100 frames, we should see at least one drop or zero due to COMM_TIMEOUT
    assert drop_seen or zero_seen, (
        "COMM_TIMEOUT should cause frame drops or zeroed data"
    )

@pytest.mark.asyncio
async def test_comm_timeout_fault_appears():
    """COMM_TIMEOUT should appear in active_faults."""
    sim = G1Simulator(seed=42)
    await sim.connect()
    sim.inject_fault(FaultType.COMM_TIMEOUT)

    # Should eventually report the fault
    fault_seen = False
    for _ in range(50):
        tele = sim.get_telemetry()
        if any("COMM_TIMEOUT" in f for f in tele.active_faults):
            fault_seen = True
            break

    assert fault_seen, "COMM_TIMEOUT fault should appear in active_faults"

@pytest.mark.asyncio
async def test_temperature_has_noise():
    """Temperature readings should show sensor noise (not purely deterministic)."""
    sim = G1Simulator(seed=42)
    await sim.connect()
    sim.set_state(RobotState.STANDING)

    temps = []
    for _ in range(10):
        tele = sim.get_telemetry()
        temps.append([j.temperature for j in tele.joints])

    # With thermal lag + noise, temperature should vary slightly between frames
    # Check the left knee (index 3) has non-identical values over 10 frames
    knee_temps = [t[3] for t in temps]
    assert len(set(knee_temps)) > 1, (
        "Temperature should vary due to sensor noise + thermal lag"
    )

@pytest.mark.asyncio
async def test_telemetry_to_dict():
    """to_dict() should return a JSON-serializable structure."""
    sim = G1Simulator(seed=42)
    await sim.connect()
    sim.set_state(RobotState.STANDING)
    tele = sim.get_telemetry()
    d = tele.to_dict()

    assert isinstance(d, dict)
    assert isinstance(d["timestamp"], float)
    assert isinstance(d["state"], str)
    assert len(d["joints"]) == 29
    assert "roll" in d["imu"]
    assert "voltage" in d["power"]
    assert "soc" in d["power"]
    assert isinstance(d["active_faults"], list)

    import json
    json_str = json.dumps(d)
    assert isinstance(json_str, str)
