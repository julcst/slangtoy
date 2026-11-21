from pathlib import Path

import slangpy as spy

from slangtoy import Frame, SlangToy


class SimpleWater(SlangToy):
    def __init__(self) -> None:
        super().__init__(Path(__file__).with_suffix(".slang"), title="Simple Water")
        self.field: list[spy.Tensor] | None = None

    def realloc(self, frame: Frame) -> None:
        shape = (frame.target.height, frame.target.width)  # Python uses (height, width)
        if self.field is None or tuple(self.field[0].shape) != shape:
            self.field = [
                spy.Tensor.empty(self.device, shape, dtype=float),
                spy.Tensor.empty(self.device, shape, dtype=float),
            ]

    def render(self, frame: Frame) -> None:
        self.realloc(frame)

        prev, curr = self.field[frame.index % 2], self.field[(frame.index + 1) % 2]

        height, width = prev.shape
        x, y = self.mouse_pos or (width // 2, height // 2)
        if 0 <= x < width and 0 <= y < height:
            cursor = prev.cursor()
            cursor[y * width + x].write(spy.math.sin(frame.time * 10.0))
            cursor.apply()

        self.module.waveprop(
            spy.grid(curr.shape),
            prev,
            curr,
            frame.dt * 0.01,
            _append_to=frame.encoder,
        )
        self.module.shade(
            spy.call_id(2), curr, _result=frame.target, _append_to=frame.encoder
        )


if __name__ == "__main__":
    SimpleWater().run()
