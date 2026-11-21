from pathlib import Path

import slangpy as spy

from slangtoy import Frame, SlangToy


class KernelDrift(SlangToy):
    def __init__(self) -> None:
        super().__init__(
            Path(__file__).with_suffix(".slang"), title="Kernel Drift"
        )

    def render(self, frame: Frame) -> None:
        self.module.render(
            spy.call_id(2), _result=frame.target, _append_to=frame.encoder
        )


if __name__ == "__main__":
    KernelDrift().run()
