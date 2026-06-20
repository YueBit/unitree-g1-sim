#!/usr/bin/env python3
"""Fault injection example — inject joint overtemperature and detect it."""

import asyncio
from unitree_g1_sim import G1Simulator, RobotState, FaultType


async def main():
    sim = G1Simulator(seed=42)
    await sim.connect()
    sim.set_state(RobotState.STANDING)

    # Run without faults
    print("=== Normal operation ===")
    for _ in range(3):
        tele = sim.get_telemetry()
        left_knee = tele.joints[3]
        print(
            f"  left_knee: temp={left_knee.temperature:.1f}°C  "
            f"faults={tele.active_faults or 'none'}"
        )
        await asyncio.sleep(0.1)

    # Inject joint overtemperature
    print("\n=== JOINT_OVERTEMP injected ===")
    sim.inject_fault(FaultType.JOINT_OVERTEMP)

    for i in range(15):
        tele = sim.get_telemetry()
        left_knee = tele.joints[3]
        right_knee = tele.joints[9]
        print(
            f"  t={i:2d}  L-knee={left_knee.temperature:5.1f}°C  "
            f"R-knee={right_knee.temperature:5.1f}°C  "
            f"faults={tele.active_faults}"
        )
        await asyncio.sleep(0.1)

    # Clear and verify
    print("\n=== Faults cleared ===")
    sim.clear_faults()
    tele = sim.get_telemetry()
    print(f"  faults={tele.active_faults or 'none'}")
    print(f"  left_knee temp={tele.joints[3].temperature:.1f}°C")

    await sim.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
