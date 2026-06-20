#!/usr/bin/env python3
"""Run built-in diagnostic tests."""

import asyncio
from unitree_g1_sim import G1Simulator, FaultType
from unitree_g1_sim.builtin_tests import TEST_CATALOG


async def main():
    sim = G1Simulator(seed=42)
    await sim.connect()

    # Inject a fault to demonstrate diagnostic detection
    sim.inject_fault(FaultType.JOINT_OVERTEMP)

    for tc_id, entry in TEST_CATALOG.items():
        result = await entry["fn"](sim)
        status_icon = {"PASS": "✅", "FAIL": "❌", "WARNING": "⚠️"}.get(
            result.status.value, "?"
        )
        print(f"{status_icon} {tc_id} {result.test_name}: {result.summary}")
        print(f"   ({result.duration_s:.1f}s, {len(result.steps)} steps)")
        print()

    await sim.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
