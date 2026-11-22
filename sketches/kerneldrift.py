from pathlib import Path

import numpy as np
import slangpy as spy

from slangtoy import Frame, SlangToy

from imgui_bundle import imgui

CHANNELS = 4


class KernelDrift(SlangToy):
    def __init__(self) -> None:
        super().__init__(Path(__file__).with_suffix(".slang"), title="Kernel Drift")
        self.field: list[spy.Tensor] | None = None
        self.palette = [
            spy.float3(0.0, 0.0, 0.0),
            spy.float3(1.0, 0.0, 0.0),
            spy.float3(0.0, 1.0, 0.0),
            spy.float3(0.0, 0.0, 1.0),
        ]
        self.pull = 0.01
        self.momentum = 0.9
        self.reset_kernel()

    def reset_kernel(self) -> None:
        self.kernel = np.random.randn(3, 3, CHANNELS, CHANNELS).astype(np.float32)
        self.kernel_velocity = np.zeros_like(self.kernel)

    def reset_field(self, shape: tuple[int, int]) -> None:
        # Fill with random 4-dim unit vectors
        data = np.random.randn(*shape, CHANNELS).astype(np.float32)
        data /= np.linalg.norm(data, axis=-1, keepdims=True)
        self.field = [
            spy.Tensor.empty(self.device, shape, dtype=spy.float4) for _ in range(2)
        ]
        for field in self.field:
            field.copy_from_numpy(data)

    def realloc(self, frame: Frame) -> None:
        shape = (frame.target.height, frame.target.width)  # Python uses (height, width)
        if self.field is None or tuple(self.field[0].shape) != shape:
            self.reset_field(shape)

    def kernel_drift(self) -> None:
        noise = np.random.randn(*self.kernel.shape).astype(np.float32)
        # Keep the long-run step variance steady as momentum changes.
        self.kernel_velocity *= self.momentum
        self.kernel_velocity += (1.0 - self.momentum**2) ** 0.5 * noise
        self.kernel += self.kernel_velocity - self.pull * self.kernel

    def draw_ui(self) -> None:
        imgui.begin("Params")
        _, self.pull = imgui.slider_float("Pull", self.pull, 0.0, 0.5)
        _, self.momentum = imgui.slider_float("Momentum", self.momentum, 0.0, 0.99)
        if imgui.button("Reset Kernel"):
            self.reset_kernel()
        if imgui.button("Reset Field"):
            self.reset_field(self.field[0].shape)
        imgui.end()

    def render(self, frame: Frame) -> None:
        self.realloc(frame)
        curr, old = self.field[frame.index % 2], self.field[(frame.index + 1) % 2]

        self.kernel_drift()
        shader_kernel = [
            [spy.float4x4(matrix) for matrix in row] for row in self.kernel
        ]

        self.module.convolve(
            spy.grid(curr.shape),
            curr,
            old,
            shader_kernel,
            _append_to=frame.encoder,
        )
        self.module.shade(
            old, self.palette, _result=frame.target, _append_to=frame.encoder
        )


if __name__ == "__main__":
    KernelDrift().run()
