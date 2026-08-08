"""Inspect one family from public API coverage through executable scale defaults."""

from __future__ import annotations

from silva_networks import (
    SILVADataLoaderConfig,
    implementation_cases,
    runtime_for_tier,
    silva_family_guide,
    silva_reproduction_spec,
    silva_scaling_defaults,
)

family = "fno_deq"
case = next(item for item in implementation_cases() if item.key == "recent_equilibrium_families")
guide = silva_family_guide(family)
reproduction = silva_reproduction_spec(family)
defaults = silva_scaling_defaults(family, tier="smoke")
runtime = runtime_for_tier("smoke")
loader = SILVADataLoaderConfig(batch_size=4, workers=0)

print("family", family)
print("public objects", len(case.public_objects))
print("verification", reproduction.verification_level)
print("benchmark tasks", len(guide.benchmark_tasks))
print("solver", defaults["config"].solver)
print("max iterations", defaults["config"].max_iter)
print("runtime", runtime.device, runtime.mixed_precision)
print("loader", loader.batch_size, loader.workers)
