# SlangToy

A collection of small interactive Slang shader sketches.

```sh
uv run sketch --list
uv run sketch new my_sketch
uv run sketch my_sketch
```

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
