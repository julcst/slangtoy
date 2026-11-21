# SlangToy

Run these commands from the project directory.

On the first run, uv builds ImGui Bundle from source with its window backend
disabled. This requires a C++ toolchain (Xcode Command Line Tools on macOS).
If an existing environment still has the prebuilt ImGui Bundle wheel, replace
it once with `uv sync --reinstall-package imgui-bundle`.

```sh
uv run sketch --list
uv run sketch water
uv run sketch new my_sketch
uv run sketch my_sketch
```

Each sketch lives in `sketches/` as a matching `.py` and `.slang` pair.
The `new` command creates both files, sets the window title from the name, and
wires up a black starter shader. Set up resources in `setup()`, build controls
with `imgui_bundle.imgui` in `draw_ui()`, and record shader commands in
`render(frame)`. Sketches share `Frame` and `SlangToy` from the library in
`src/slangtoy/`.

## Shortcuts

| Key | Action |
| --- | --- |
| **Space** | Pause or resume shader rendering. The UI stays interactive. |
| **I** | Freeze the current shader frame, save a linear RGBA OpenEXR, and open it in tev. Press **Space** to resume. |
| **S** | Save a PNG screenshot of the shader image. |
| **O** | Toggle the FPS and GPU timing overlay. |
| **L** | Manually reload the shader when hot reload is enabled. |
| **−** / **=** | Decrease or increase the ImGui font and control sizes. |
| **Esc** | Close the sketch. |
| **R** | Reset the field in Square World. |

Shortcuts apply while the sketch has focus and ImGui is not using the keyboard.
PNG and EXR captures are saved in the working directory without UI. The EXR
preserves HDR values, NaNs, and infinities from the shader target. SlangToy
looks for [tev](https://github.com/Tom94/tev) on `PATH` or in the standard
macOS Applications folders. The terminal prints the EXR path so you can open
it manually. In tev, scroll to zoom, hover over pixels to see channel values,
and press **H** for its shortcuts.

## Square World

Run with `uv run sketch square_world` or `uv run slangtoy`. Changes to the
simulation controls take effect on the next rendered frame.

The **Simulation** panel sets the advance rate and collapse chance per second.
**Steps per frame** divides the elapsed frame time into smaller updates. More
steps increase GPU work while keeping the advance budget and cumulative
collapse chance for the same elapsed time. **Reset field** draws independent
standard-normal values. **Reseed kernel** changes how the current field evolves.
**Randomize palette** draws new display colors.

A random kernel mixes signed values across a wrapping grid. **Probability**
switches the current field between **Root**, which uses normalized squared
amplitudes, and **Logit**, which uses softmax. An all-zero cell uses equal
probabilities in both modes. These probabilities drive collapse sampling and
palette blending. Collapse gives the sampled channel 99% before the next
normalization pass and divides the rest equally. Root mode also preserves the
signs of nonzero amplitudes during collapse.

**Normalize energy** applies one scale factor to the entire field before
collapse sampling and after each step. The target is the sum of squares across
all cells and channels in the most recent reset field.
