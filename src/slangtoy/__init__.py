# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path

import slangpy as spy

from .slangtoy import Frame, SlangToy


class Demo(SlangToy):
    def render(self, frame: Frame) -> None:
        self.module.render(
            spy.call_id(2),
            frame.resolution,
            _result=frame.target,
            _append_to=frame.encoder,
        )


def main() -> None:
    Demo(shader=Path(__file__).with_name("draw.slang")).run()
