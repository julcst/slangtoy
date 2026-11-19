# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections.abc import Callable
from colorsys import hsv_to_rgb
from pathlib import Path

import numpy as np
import slangpy as spy

from .slangtoy import Frame, SlangToy

RES = 512
TILE = 8
PALETTE = np.array([
    (0.019, 0.025, 0.095),
    (0.028, 0.51, 0.39),
    (0.94, 0.18, 0.09),
    (0.98, 0.72, 0.24),
], dtype=np.float32)


class Demo(SlangToy):
    def __init__(self) -> None:
        super().__init__(Path(__file__).with_name("draw.slang"), title="Logit World")
        self.rng = np.random.default_rng()
        self.pending: list[Callable[[], None]] = []
        self.fields = tuple(self.buffer(RES, RES, len(PALETTE)) for _ in range(3))
        self.kernel = self.buffer(8, len(PALETTE), len(PALETTE))
        self.palette = spy.Tensor.from_numpy(self.device, PALETTE)
        self.sums = []
        height, width, channels = self.fields[0].shape
        while True:
            height = (height + TILE - 1) // TILE
            width = (width + TILE - 1) // TILE
            self.sums.append(self.buffer(height, width, channels))
            if height == width == 1:
                break
        self.reset_field()
        self.reseed_kernel()

    def buffer(self, *shape: int) -> spy.Tensor:
        return spy.Tensor.empty(self.device, shape, dtype=float)

    def setup(self) -> None:
        panel = spy.ui.Window(
            self.ui.screen, "Simulation",
            position=spy.float2(10, 80), size=spy.float2(300, 200),
        )
        self.normalization = spy.ui.CheckBox(panel, "Normalize energy", value=True)
        self.collapse_rate = spy.ui.SliderFloat(
            panel, "Collapse rate", value=0.04, min=0, max=5, format="%.3f%%",
            flags=spy.ui.SliderFlags.logarithmic | spy.ui.SliderFlags.always_clamp,
        )
        self.advance_rate = spy.ui.SliderFloat(
            panel, "Advance rate", value=0.2, min=0, max=1, format="%.3f",
            flags=spy.ui.SliderFlags.always_clamp,
        )
        self.steps_per_frame = spy.ui.SliderInt(
            panel, "Steps per frame", value=1, min=1, max=16,
            flags=spy.ui.SliderFlags.always_clamp,
        )
        spy.ui.Button(panel, "Reset field", lambda: self.pending.append(self.reset_field))
        spy.ui.Button(panel, "Reseed kernel", lambda: self.pending.append(self.reseed_kernel))
        spy.ui.Button(panel, "Randomize palette", lambda: self.pending.append(self.randomize_palette))

    def reset_field(self) -> None:
        field = self.fields[0]
        field.copy_from_numpy(self.rng.standard_normal(tuple(field.shape), dtype=np.float32))
        self.seed = int(self.rng.integers(2**31))

    def reseed_kernel(self) -> None:
        weights = self.rng.normal(0, 0.30, tuple(self.kernel.shape))
        self.kernel.copy_from_numpy(weights.astype(np.float32))

    def randomize_palette(self) -> None:
        channels = self.palette.shape[0]
        hues = (self.rng.random() + np.arange(channels) / channels) % 1.0
        colors = [
            hsv_to_rgb(hue, self.rng.uniform(0.65, 0.95), self.rng.uniform(0.45, 1.0))
            for hue in self.rng.permutation(hues)
        ]
        self.palette.copy_from_numpy(np.asarray(colors, dtype=np.float32) ** 2.2)

    def energy(self, field: spy.Tensor, encoder: spy.CommandEncoder) -> spy.Tensor:
        for i, result in enumerate(self.sums):
            self.module.sumBlock(
                spy.call_id(3), field, i == 0, _result=result, _append_to=encoder
            )
            field = result
        return field

    def on_key(self, event: spy.KeyboardEvent) -> None:
        if event.is_key_press() and event.key == spy.KeyCode.r:
            self.pending.append(self.reset_field)

    def render(self, frame: Frame) -> None:
        for action in self.pending:
            action()
        self.pending.clear()

        current, next_field, scratch = self.fields
        encoder = frame.encoder
        seed = self.seed
        if not self.paused:
            steps = self.steps_per_frame.value
            cell = spy.call_id(3)
            normalize = self.normalization.value
            advance_rate = self.advance_rate.value / steps
            # Split the collapse probability by compounding the probability of no collapse.
            collapse_rate = 1 - (1 - self.collapse_rate.value / 100) ** (1 / steps)
            for _ in range(steps):
                self.module.advance(
                    cell, current, self.kernel, advance_rate,
                    _result=next_field, _append_to=encoder,
                )
                if collapse_rate > 0:
                    energy = self.energy(next_field, encoder) if normalize else self.sums[-1]
                    self.module.collapse(
                        cell, next_field, energy, normalize, seed, collapse_rate,
                        _result=scratch, _append_to=encoder,
                    )
                    next_field, scratch = scratch, next_field
                if normalize:
                    self.module.normalize(
                        cell, next_field, self.energy(next_field, encoder),
                        _result=scratch, _append_to=encoder,
                    )
                    next_field, scratch = scratch, next_field
                current, next_field, scratch = next_field, scratch, current
                seed = (seed + 1) % 2**31

        self.module.render(
            spy.call_id(2), frame.resolution, current, self.palette,
            _result=frame.target, _append_to=encoder,
        )
        self.fields = (current, next_field, scratch)
        self.seed = seed


def main() -> None:
    Demo().run()
