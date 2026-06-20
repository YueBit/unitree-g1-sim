"""
Built-in diagnostic test cases for Unitree G1 robot.

Covers: joint range of motion, temperature, IMU calibration,
standing stability, and power consumption.

Each test returns a structured ``TestResult`` with measurements,
thresholds, and pass/fail/warning verdicts.
"""

import asyncio
import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from .config import G1_JOINTS, G1_SPECS
from .simulator import G1Simulator, RobotState


class TestStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    SKIP = "SKIP"
    RUNNING = "RUNNING"


@dataclass
class TestStep:
    name: str
    status: TestStatus
    measured: Any
    threshold: Any
    unit: str
    detail: str = ""


@dataclass
class TestResult:
    test_id: str
    test_name: str
    status: TestStatus
    duration_s: float
    steps: list[TestStep]
    raw_data: dict[str, Any]
    summary: str


# ═══════════════════════════════════════════════════════════════════
#  TC-001: Joint Range of Motion
# ═══════════════════════════════════════════════════════════════════


async def test_joint_range_of_motion(sim: G1Simulator) -> TestResult:
    """Verify each joint stays within its hardware limits during TESTING state."""
    start = time.time()
    steps: list[TestStep] = []
    raw: dict[str, Any] = {"joints": {}}

    sim.set_state(RobotState.TESTING)
    await asyncio.sleep(0.3)

    samples = []
    for _ in range(5):
        tele = sim.get_telemetry()
        samples.append(tele.joints)
        await asyncio.sleep(0.1)

    fail_count = 0
    warn_count = 0
    for j_config in G1_JOINTS:
        lo, hi = j_config["limits"]
        name = j_config["name"]
        positions = [s[j_config["id"]].position for s in samples]
        max_pos = max(abs(p) for p in positions)
        limit_max = max(abs(lo), abs(hi))
        ratio = max_pos / limit_max if limit_max > 0 else 0
        raw["joints"][name] = {"positions": positions, "limit_ratio": ratio}

        over_limit = any(p < lo - 0.05 or p > hi + 0.05 for p in positions)
        if over_limit:
            status = TestStatus.FAIL
            fail_count += 1
            detail = "超出关节限位！"
        elif ratio > 0.95:
            status = TestStatus.WARNING
            warn_count += 1
            detail = f"接近限位边界 ({ratio:.1%})"
        else:
            status = TestStatus.PASS
            detail = "正常"

        steps.append(
            TestStep(
                name=f"[{name}] 位置限位检查",
                status=status,
                measured=round(max_pos, 3),
                threshold=round(limit_max, 3),
                unit="rad",
                detail=detail,
            )
        )

    duration = time.time() - start
    if fail_count > 0:
        overall = TestStatus.FAIL
        summary = f"❌ {fail_count}个关节超出限位，{warn_count}个接近边界"
    elif warn_count > 0:
        overall = TestStatus.WARNING
        summary = f"⚠️ {warn_count}个关节接近限位边界，建议检查"
    else:
        overall = TestStatus.PASS
        summary = f"✅ 全部{len(G1_JOINTS)}个关节位置正常"

    return TestResult(
        test_id="TC-001",
        test_name="关节运动范围测试",
        status=overall,
        duration_s=round(duration, 2),
        steps=steps,
        raw_data=raw,
        summary=summary,
    )


# ═══════════════════════════════════════════════════════════════════
#  TC-002: Joint Temperature
# ═══════════════════════════════════════════════════════════════════


async def test_joint_temperature(sim: G1Simulator) -> TestResult:
    """Verify all joint temperatures are within safe limits."""
    start = time.time()
    steps: list[TestStep] = []
    raw: dict[str, Any] = {"temperatures": {}}

    sim.set_state(RobotState.STANDING)
    await asyncio.sleep(0.5)

    tele = sim.get_telemetry()
    fail_count = 0
    warn_count = 0

    for j in tele.joints:
        raw["temperatures"][j.name] = j.temperature
        warn_t = G1_SPECS["warning_joint_temp_celsius"]
        max_t = G1_SPECS["max_joint_temp_celsius"]

        if j.temperature >= max_t:
            status = TestStatus.FAIL
            fail_count += 1
            detail = f"超过最大温度限制 {max_t}°C！"
        elif j.temperature >= warn_t:
            status = TestStatus.WARNING
            warn_count += 1
            detail = f"接近温度警戒值 {warn_t}°C"
        else:
            status = TestStatus.PASS
            detail = "温度正常"

        steps.append(
            TestStep(
                name=f"[{j.name}] 温度",
                status=status,
                measured=j.temperature,
                threshold=max_t,
                unit="°C",
                detail=detail,
            )
        )

    duration = time.time() - start
    if fail_count > 0:
        overall = TestStatus.FAIL
        summary = f"❌ {fail_count}个关节过温，立即停机检查"
    elif warn_count > 0:
        overall = TestStatus.WARNING
        summary = f"⚠️ {warn_count}个关节温度偏高"
    else:
        overall = TestStatus.PASS
        summary = (
            f"✅ 全部关节温度正常"
            f"（最高 {max(j.temperature for j in tele.joints):.1f}°C）"
        )

    return TestResult(
        test_id="TC-002",
        test_name="关节温度检测",
        status=overall,
        duration_s=round(duration, 2),
        steps=steps,
        raw_data=raw,
        summary=summary,
    )


# ═══════════════════════════════════════════════════════════════════
#  TC-003: IMU Calibration
# ═══════════════════════════════════════════════════════════════════


async def test_imu_calibration(sim: G1Simulator) -> TestResult:
    """Verify IMU bias and noise while standing still."""
    start = time.time()
    steps: list[TestStep] = []
    raw: dict[str, Any] = {"imu_samples": []}

    sim.set_state(RobotState.STANDING)
    await asyncio.sleep(0.2)

    samples = []
    for _ in range(20):
        tele = sim.get_telemetry()
        imu = tele.imu
        samples.append(
            {
                "roll": imu.roll,
                "pitch": imu.pitch,
                "yaw": imu.yaw,
                "gyro_x": imu.gyro_x,
                "gyro_y": imu.gyro_y,
                "gyro_z": imu.gyro_z,
                "accel_z": imu.accel_z,
            }
        )
        await asyncio.sleep(0.05)

    raw["imu_samples"] = samples

    rolls = [s["roll"] for s in samples]
    pitches = [s["pitch"] for s in samples]
    gyros = [
        math.sqrt(s["gyro_x"] ** 2 + s["gyro_y"] ** 2 + s["gyro_z"] ** 2)
        for s in samples
    ]
    accel_z = [s["accel_z"] for s in samples]

    roll_std = float(np.std(rolls))
    pitch_std = float(np.std(pitches))
    gyro_bias = float(np.mean(gyros))
    accel_error = abs(float(np.mean(accel_z)) - 9.81)

    checks = [
        ("姿态Roll抖动 (σ)", roll_std, 0.05, "rad", roll_std < 0.05),
        ("姿态Pitch抖动 (σ)", pitch_std, 0.05, "rad", pitch_std < 0.05),
        ("陀螺仪偏置", gyro_bias, 0.02, "rad/s", gyro_bias < 0.02),
        ("加速度计Z轴误差", accel_error, 0.1, "m/s²", accel_error < 0.1),
    ]

    fail_count = 0
    for name, measured, threshold, unit, ok in checks:
        status = TestStatus.PASS if ok else TestStatus.FAIL
        if not ok:
            fail_count += 1
        steps.append(
            TestStep(
                name=f"[IMU] {name}",
                status=status,
                measured=round(measured, 5),
                threshold=threshold,
                unit=unit,
                detail="正常" if ok else "超出校准阈值，需重新标定",
            )
        )

    overall = TestStatus.FAIL if fail_count > 0 else TestStatus.PASS
    summary = (
        f"❌ IMU校准异常，{fail_count}项超标"
        if fail_count > 0
        else "✅ IMU校准正常，姿态估计精度良好"
    )

    return TestResult(
        test_id="TC-003",
        test_name="IMU校准验证",
        status=overall,
        duration_s=round(time.time() - start, 2),
        steps=steps,
        raw_data=raw,
        summary=summary,
    )


# ═══════════════════════════════════════════════════════════════════
#  TC-004: Standing Stability
# ═══════════════════════════════════════════════════════════════════


async def test_standing_stability(sim: G1Simulator) -> TestResult:
    """Check ZMP stability margins while standing."""
    start = time.time()
    steps: list[TestStep] = []
    raw: dict[str, Any] = {"stability_data": []}

    sim.set_state(RobotState.STANDING)
    await asyncio.sleep(0.3)

    roll_maxes = []
    pitch_maxes = []
    for _ in range(30):
        tele = sim.get_telemetry()
        roll_maxes.append(abs(tele.imu.roll))
        pitch_maxes.append(abs(tele.imu.pitch))
        raw["stability_data"].append(
            {
                "roll": tele.imu.roll,
                "pitch": tele.imu.pitch,
                "t": tele.timestamp,
            }
        )
        await asyncio.sleep(0.05)

    max_roll = max(roll_maxes)
    max_pitch = max(pitch_maxes)
    avg_roll = float(np.mean(roll_maxes))
    avg_pitch = float(np.mean(pitch_maxes))

    max_roll_deg = math.degrees(max_roll)
    max_pitch_deg = math.degrees(max_pitch)

    ROLL_THRESHOLD_DEG = 5.0
    PITCH_THRESHOLD_DEG = 5.0

    results = [
        (
            "最大横滚角",
            max_roll_deg,
            ROLL_THRESHOLD_DEG,
            "°",
            max_roll_deg < ROLL_THRESHOLD_DEG,
        ),
        (
            "最大俯仰角",
            max_pitch_deg,
            PITCH_THRESHOLD_DEG,
            "°",
            max_pitch_deg < PITCH_THRESHOLD_DEG,
        ),
        ("平均横滚偏差", math.degrees(avg_roll), 2.0, "°", math.degrees(avg_roll) < 2.0),
        (
            "平均俯仰偏差",
            math.degrees(avg_pitch),
            2.0,
            "°",
            math.degrees(avg_pitch) < 2.0,
        ),
    ]

    fail_count = 0
    for name, measured, threshold, unit, ok in results:
        if not ok:
            fail_count += 1
        steps.append(
            TestStep(
                name=f"[稳定性] {name}",
                status=TestStatus.PASS if ok else TestStatus.FAIL,
                measured=round(measured, 2),
                threshold=threshold,
                unit=unit,
                detail="稳定" if ok else "超出稳定裕度",
            )
        )

    overall = TestStatus.FAIL if fail_count > 0 else TestStatus.PASS
    summary = (
        f"❌ 站立稳定性不足，{fail_count}项超标"
        if fail_count > 0
        else "✅ 站立稳定性良好"
    )

    return TestResult(
        test_id="TC-004",
        test_name="站立稳定性测试",
        status=overall,
        duration_s=round(time.time() - start, 2),
        steps=steps,
        raw_data=raw,
        summary=summary,
    )


# ═══════════════════════════════════════════════════════════════════
#  TC-005: Power Consumption
# ═══════════════════════════════════════════════════════════════════


async def test_power_consumption(sim: G1Simulator) -> TestResult:
    """Measure current draw and battery health across operating states."""
    start = time.time()
    steps: list[TestStep] = []
    raw: dict[str, Any] = {"power_readings": {}}

    scenarios = [
        (RobotState.STANDBY, "待机", 5.0, 0.5),
        (RobotState.STANDING, "站立", 15.0, 2.0),
        (RobotState.WALKING, "行走", 35.0, 5.0),
    ]

    for state, label, max_power_a, duration in scenarios:
        sim.set_state(state)
        await asyncio.sleep(duration * 0.2)

        readings = []
        for _ in range(5):
            tele = sim.get_telemetry()
            readings.append(
                {
                    "current": tele.power.current,
                    "voltage": tele.power.voltage,
                    "power_w": tele.power.power_w,
                }
            )
            await asyncio.sleep(0.05)

        avg_current = float(np.mean([r["current"] for r in readings]))
        avg_power = float(np.mean([r["power_w"] for r in readings]))
        raw["power_readings"][label] = readings

        ok = avg_current < max_power_a
        steps.append(
            TestStep(
                name=f"[功耗] {label}电流",
                status=TestStatus.PASS if ok else TestStatus.FAIL,
                measured=round(avg_current, 2),
                threshold=max_power_a,
                unit="A",
                detail=f"平均功率 {avg_power:.1f}W"
                + ("" if ok else " — 超出额定值"),
            )
        )

    # battery voltage check
    tele = sim.get_telemetry()
    voltage_ok = tele.power.voltage >= G1_SPECS["battery_voltage_min"]
    steps.append(
        TestStep(
            name="[功耗] 电池电压",
            status=TestStatus.PASS if voltage_ok else TestStatus.FAIL,
            measured=tele.power.voltage,
            threshold=G1_SPECS["battery_voltage_min"],
            unit="V",
            detail=f"SOC {tele.power.soc:.1f}%",
        )
    )

    fail_count = sum(1 for s in steps if s.status == TestStatus.FAIL)
    overall = TestStatus.FAIL if fail_count > 0 else TestStatus.PASS
    summary = (
        f"❌ {fail_count}项功耗测试不通过"
        if fail_count > 0
        else f"✅ 功耗测试通过，当前SOC {tele.power.soc:.1f}%"
    )

    return TestResult(
        test_id="TC-005",
        test_name="功耗测试",
        status=overall,
        duration_s=round(time.time() - start, 2),
        steps=steps,
        raw_data=raw,
        summary=summary,
    )


# ═══════════════════════════════════════════════════════════════════
#  Test catalog
# ═══════════════════════════════════════════════════════════════════

TEST_CATALOG: dict[str, dict] = {
    "TC-001": {
        "fn": test_joint_range_of_motion,
        "name": "关节运动范围测试",
        "category": "机械",
        "duration_estimate": 2,
    },
    "TC-002": {
        "fn": test_joint_temperature,
        "name": "关节温度检测",
        "category": "热管理",
        "duration_estimate": 3,
    },
    "TC-003": {
        "fn": test_imu_calibration,
        "name": "IMU校准验证",
        "category": "传感器",
        "duration_estimate": 2,
    },
    "TC-004": {
        "fn": test_standing_stability,
        "name": "站立稳定性测试",
        "category": "控制",
        "duration_estimate": 4,
    },
    "TC-005": {
        "fn": test_power_consumption,
        "name": "功耗测试",
        "category": "电源",
        "duration_estimate": 5,
    },
}
