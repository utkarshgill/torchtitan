"""muP helpers shared across path vision backbones (convnext, fastvit).

muP scales hidden/readout init to 1/sqrt(fan_in) so pre-activations stay O(1) as width grows, pairs that
with a per-group lr eta/m and a 1/m readout multiplier (m = width / base_width). Width-independent layers
(depthwise convs, the input stem) are left at the standard init and lr -- their fan_in does not grow.
"""
from __future__ import annotations

BASE_WIDTH = 256


def hidden_std(fan_in: int) -> float:
    # muP init for a width-scaling matmul weight: 1/sqrt(fan_in), keeping pre-activations O(1).
    return fan_in**-0.5


def scale_dims(dims_base: tuple[int, ...], width: int, base_width: int = BASE_WIDTH) -> tuple[int, ...]:
    # Scale every stage width by m = width / base_width, keeping depths and stage ratios fixed.
    return tuple(d * width // base_width for d in dims_base)
