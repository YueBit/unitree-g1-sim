#!/usr/bin/env python3
"""Basic usage example — connect to the simulator and read telemetry."""

import asyncio
from unitree_g1_sim import G1Simulator, RobotState


async def main():
    sim = G1Simulator(seed=42)

    # connect (state: OFFLINE → INITIALIZING → STANDBY)
    await sim.connect()
    print(f"State: {sim.state}")

    # stand up
    sim.set_state(RobotState.STANDING)
    await asyncio.sleep(0.3)

    # read one telemetry frame
    tele = sim.get_telemetry()
    print(f"\nJoints: {len(tele.joints)} (first 3 shown)")
    for j in tele.joints[:3]:
        print(f"  {j.name:25s}  pos={j.position:7.4f}  temp={j.temperature:5.1f}°C")

    print(f"\nIMU:   roll={tele.imu.roll:.5f}  pitch={tele.imu.pitch:.5f}")
    print(f"Power: {tele.power.soc:.1f}% SOC  {tele.power.voltage:.1f}V")
    print(f"Faults: {tele.active_faults or 'none'}")

    await sim.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
