# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections.abc import Callable
from colorsys import hsv_to_rgb
from math import expm1, log1p
from pathlib import Path

import numpy as np
import slangpy as spy
from imgui_bundle import imgui

from slangtoy import Frame, SlangToy

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
        super().__init__(Path(__file__).with_suffix(".slang"), title="Square World")
        self.rng = np.random.default_rng()
        self.pending: list[Callable[[], None]] = []
        self.probability_mode = 1
        self.normalization = True
        self.collapse_rate = 1.31
        self.advance_rate = 12.0
        self.steps_per_frame = 1
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

    def draw_ui(self) -> None:
        font_size = imgui.get_font_size()
        imgui.set_next_window_pos((font_size, font_size), imgui.Cond_.first_use_ever)
        visible, _ = imgui.begin("Simulation", flags=imgui.WindowFlags_.always_auto_resize)
        if visible:
            clamp = imgui.SliderFlags_.always_clamp
            imgui.push_item_width(10 * font_size)
            _, self.probability_mode = imgui.combo("Probability", self.probability_mode, ["Logit", "Root"])
            _, self.normalization = imgui.checkbox("Normalize energy", self.normalization)
            _, self.collapse_rate = imgui.slider_float(
                "Collapse chance / s", self.collapse_rate, 0, 95, "%.3f%%",
                flags=imgui.SliderFlags_.logarithmic | clamp,
            )
            _, self.advance_rate = imgui.slider_float(
                "Advance rate / s", self.advance_rate, 0, 60, "%.3f", flags=clamp,
            )
            _, self.steps_per_frame = imgui.slider_int(
                "Steps per frame", self.steps_per_frame, 1, 16, flags=clamp,
            )
            imgui.pop_item_width()
            if imgui.button("Reset field"):
                self.pending.append(self.reset_field)
            if imgui.button("Reseed kernel"):
                self.pending.append(self.reseed_kernel)
            if imgui.button("Randomize palette"):
                self.pending.append(self.randomize_palette)
        imgui.end()

    def reset_field(self) -> None:
        field = self.fields[0]
        initial = self.rng.standard_normal(tuple(field.shape), dtype=np.float32)
        field.copy_from_numpy(initial)
        self.initial_energy = float(np.square(initial, dtype=np.float64).sum())
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
        root = self.probability_mode == 1
        if not self.paused and frame.dt > 0:
            steps = self.steps_per_frame
            cell = spy.call_id(3)
            normalize = self.normalization
            step_dt = frame.dt / steps
            advance_rate = self.advance_rate * step_dt
            # Compound the probability of no collapse over the elapsed time.
            collapse_rate = -expm1(log1p(-self.collapse_rate / 100) * step_dt)
            for _ in range(steps):
                self.module.advance(
                    cell, current, self.kernel, advance_rate,
                    _result=next_field, _append_to=encoder,
                )
                if collapse_rate > 0:
                    energy = self.energy(next_field, encoder) if normalize else self.sums[-1]
                    self.module.collapse(
                        cell, next_field, energy, self.initial_energy, normalize, root, seed, collapse_rate,
                        _result=scratch, _append_to=encoder,
                    )
                    next_field, scratch = scratch, next_field
                if normalize:
                    self.module.normalize(
                        cell, next_field, self.energy(next_field, encoder), self.initial_energy,
                        _result=scratch, _append_to=encoder,
                    )
                    next_field, scratch = scratch, next_field
                current, next_field, scratch = next_field, scratch, current
                seed = (seed + 1) % 2**31

        self.module.render(
            spy.call_id(2), frame.resolution, current, self.palette, root,
            _result=frame.target, _append_to=encoder,
        )
        self.fields = (current, next_field, scratch)
        self.seed = seed


def main() -> None:
    Demo().run()


if __name__ == "__main__":
    main()
