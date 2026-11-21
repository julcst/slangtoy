# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""A SlangPy window for experimenting with interactive shaders."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from math import floor
from pathlib import Path
from time import perf_counter, sleep

import slangpy as spy
from imgui_bundle import imgui
from slangpy.ui import imgui_bundle as imgui_backend

_TIMING_SLOTS = 4
_LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Frame:
    """Render inputs for one frame. Time and dt are in seconds.

    Time and index advance only when rendering. Record commands during render
    and let the window handle submission.
    """

    target: spy.Texture
    encoder: spy.CommandEncoder
    index: int
    time: float
    dt: float

    @property
    def resolution(self) -> spy.int2:
        return spy.int2(self.target.width, self.target.height)


class SlangToy:
    """Override render() to draw and draw_ui() for immediate-mode ImGui controls.

    Pass a shader path to expose a hot-reloadable SlangPy module as self.module.
    Use the frame's encoder for all drawing to include it in GPU timings.
    Read self.mouse_pos for the last pointer position as (x, y) render pixels
    from the top left, or None before the first mouse event.
    """

    def __init__(
        self,
        shader: str | Path | None = None,
        *,
        title: str = "SlangToy",
        size: tuple[int, int] = (960, 640),
        vsync: bool = True,
        hot_reload: bool = True,
        watch: tuple[str | Path, ...] = (),
        gpu_profiling: bool = True,
        show_stats: bool = True,
    ) -> None:
        shader_path = Path(shader).resolve() if shader is not None else None
        self.window = spy.Window(width=size[0], height=size[1], title=title)
        self.device = spy.create_device(
            include_paths=[shader_path.parent] if shader_path else [],
            enable_hot_reload=False,
        )
        self.surface = self.device.create_surface(self.window)
        self.paused = False
        self.mouse_pos: tuple[int, int] | None = None
        self.show_stats = show_stats
        self.fps = 0.0
        self.gpu_ms: float | None = None
        self._vsync = vsync
        self._watch: tuple[Path, ...] = ()
        if hot_reload and shader_path is not None:
            self._watch = (shader_path, *(Path(path).resolve() for path in watch))
        self._stamps = self._shader_stamps()
        self.module = (
            spy.Module.load_from_file(self.device, str(shader_path))
            if shader_path
            else None
        )
        self._check_after = 0.0
        self._reload_requested = False
        self._timestamp_hz = self.device.info.timestamp_frequency
        self._query_pool = (
            self.device.create_query_pool(spy.QueryType.timestamp, 2 * _TIMING_SLOTS)
            if gpu_profiling
            and self._timestamp_hz > 0
            and self.device.has_feature(spy.Feature.timestamp_query)
            else None
        )
        self._timings: deque[tuple[int, int]] = deque()
        self._next_timestamp = 0
        self._screenshots: list[Path] = []
        self._inspect_requested = False
        self._ui_scale = 1.0
        self._pending_size: tuple[int, int] | None = None
        self._status = ""
        self._status_until = 0.0

    def setup(self) -> None:
        """Create custom resources once, before the first frame."""

    def draw_ui(self) -> None:
        """Build controls with imgui_bundle.imgui on each display refresh."""

    def render(self, frame: Frame) -> None:
        """Record work for the linear RGBA16F target.

        If drawing with a newly reloaded module raises an error, the same frame
        is retried with the previous module and a fresh encoder.
        """
        frame.encoder.clear_texture_float(frame.target, clear_value=(0, 0, 0, 1))

    def on_key(self, event: spy.KeyboardEvent) -> None:
        """Handle a keyboard event that the UI and shortcuts left unhandled."""

    def on_mouse(self, event: spy.MouseEvent) -> None:
        """Handle a mouse event that the UI left unhandled."""

    def screenshot(self, path: str | Path | None = None) -> Path:
        """Queue a PNG of the next displayed shader frame, without the UI."""
        if path is None:
            path = Path(f"screenshot-{datetime.now():%Y%m%d-%H%M%S-%f}.png")
        path = Path(path)
        self._screenshots.append(path)
        return path

    def _keyboard_event(self, event: spy.KeyboardEvent) -> None:
        if imgui_backend.handle_keyboard_event(event):
            return
        if event.type == spy.KeyboardEventType.key_press:
            match event.key:
                case spy.KeyCode.escape:
                    self.window.close()
                case spy.KeyCode.space:
                    self.paused = not self.paused
                case spy.KeyCode.s:
                    self.screenshot()
                case spy.KeyCode.o:
                    self.show_stats = not self.show_stats
                case spy.KeyCode.l if self._watch:
                    self._reload_requested = True
                case spy.KeyCode.i:
                    self.paused = True
                    self._inspect_requested = True
                case spy.KeyCode.minus:
                    self._ui_scale = max(0.75, self._ui_scale - 0.125)
                case spy.KeyCode.equal:
                    self._ui_scale = min(2.5, self._ui_scale + 0.125)
                case _:
                    self.on_key(event)
        else:
            self.on_key(event)

    def _mouse_event(self, event: spy.MouseEvent) -> None:
        pos = event.pos
        self.mouse_pos = floor(pos.x), floor(pos.y)
        if not imgui_backend.handle_mouse_event(event):
            self.on_mouse(event)

    def _shader_stamps(self) -> tuple[int | None, ...]:
        stamps = []
        for path in self._watch:
            try:
                stamps.append(path.stat().st_mtime_ns)
            except OSError:
                stamps.append(None)
        return tuple(stamps)

    def _report_reload(self, error: Exception | None = None) -> None:
        if error is None:
            message = "Shader reloaded"
            stream, color = sys.stdout, 32
            self._notify(message)
        else:
            message = "Shader reload failed"
            stream, color = sys.stderr, 31
            self._notify(f"{message}. See terminal.")

        headline = f"{message}: {self._watch[0].name}"
        if stream.isatty() and not os.environ.get("NO_COLOR"):
            headline = f"\x1b[{color}m{headline}\x1b[0m"
        print(headline, file=stream, flush=True)
        if error is not None:
            print(error, file=stream, flush=True)

    def _notify(self, message: str, error: Exception | None = None) -> None:
        self._status = message
        self._status_until = perf_counter() + 4.0
        if error is not None:
            _LOG.error("%s: %s", message, error)

    def _check_reload(self, now: float) -> spy.Module | None:
        if not self._watch or (now < self._check_after and not self._reload_requested):
            return None
        self._check_after = now + 0.2
        stamps = self._shader_stamps()
        if stamps == self._stamps and not self._reload_requested:
            return None
        self._stamps = stamps
        self._reload_requested = False
        try:
            # An isolated compiler session leaves the live module valid on failure.
            session = self.device.create_slang_session(
                compiler_options=self.device.desc.compiler_options
            )
            candidate = spy.Module(session.load_module(str(self._watch[0])))
        except (RuntimeError, ValueError) as error:
            self._report_reload(error)
            return None
        previous_module, self.module = self.module, candidate
        return previous_module

    def _on_resize(self, width: int, height: int) -> None:
        self._pending_size = width, height

    def _resize(self, width: int, height: int) -> None:
        config = self.surface.config
        if config is not None:
            if (width, height) == (config.width, config.height):
                return
            self.device.wait()
        if width > 0 and height > 0:
            self.surface.configure(width=width, height=height, vsync=self._vsync)
        elif config is not None:
            self.surface.unconfigure()

    def _poll_gpu(self) -> None:
        pool = self._query_pool
        if pool is None:
            return
        while self._timings and self.device.is_submit_finished(self._timings[0][1]):
            index, _ = self._timings.popleft()
            start, end = pool.get_results(index, 2)
            self.gpu_ms = (end - start) * 1_000 / self._timestamp_hz

    def _save_screenshots(self, target: spy.Texture) -> None:
        paths, self._screenshots = self._screenshots, []
        if not paths:
            return
        try:
            bitmap = target.to_bitmap().convert(
                spy.Bitmap.PixelFormat.rgb,
                spy.Bitmap.ComponentType.uint8,
                srgb_gamma=True,
            )
        except RuntimeError as error:
            self._notify("Screenshot readback failed", error)
            return
        for path in paths:
            try:
                bitmap.write(path, format=spy.Bitmap.FileFormat.png)
                self._notify(f"Saved {path.name}")
            except (OSError, RuntimeError) as error:
                self._notify(f"Could not save {path.name}", error)

    def _inspect_frame(self, target: spy.Texture) -> None:
        self._inspect_requested = False
        path = Path.cwd() / f"inspect-{datetime.now():%Y%m%d-%H%M%S-%f}.exr"
        try:
            target.to_bitmap().write(path, format=spy.Bitmap.FileFormat.exr)
        except (OSError, RuntimeError) as error:
            self._notify("Could not export inspection frame", error)
            return

        print(f"Inspection frame saved to {path}", flush=True)
        executable = shutil.which("tev")
        if executable is None and sys.platform == "darwin":
            for applications in (Path("/Applications"), Path.home() / "Applications"):
                candidate = applications / "tev.app/Contents/MacOS/tev"
                if candidate.is_file():
                    executable = str(candidate)
                    break
        if executable is None:
            self._notify("tev unavailable. EXR saved")
            return
        try:
            subprocess.Popen([executable, str(path)])
        except OSError as error:
            self._notify("Could not start tev. EXR saved", error)
        else:
            self._notify("Sent frame to tev")

    def run(self) -> None:
        """Open the window loop on the main thread and run until it closes."""
        previous_context = imgui.get_current_context()
        renderer = spy.ui.Context(self.device)
        context = imgui.create_context()
        imgui.set_current_context(context)
        io = imgui.get_io()
        io.set_ini_filename(None)
        io.backend_flags |= (
            imgui.BackendFlags_.renderer_has_textures | imgui.BackendFlags_.renderer_has_vtx_offset
        )
        io.fonts.add_font_default_vector()
        imgui.get_style().font_size_base = 16.0
        self.window.on_resize = self._on_resize
        self.window.on_keyboard_event = self._keyboard_event
        self.window.on_mouse_event = self._mouse_event
        try:
            self._resize(self.window.width, self.window.height)
            self.setup()
            self._run_frames(renderer)
        finally:
            self.window.on_resize = None
            self.window.on_keyboard_event = None
            self.window.on_mouse_event = None
            self.device.wait()
            self.surface.unconfigure()
            imgui.destroy_context(context)
            if previous_context is not None:
                imgui.set_current_context(previous_context)

    def _metrics_text(self) -> str:
        gpu = "n/a" if self._query_pool is None else "…"
        if self.gpu_ms is not None:
            gpu = f"{self.gpu_ms:.2f} ms"
        text = f"{self.fps:.1f} FPS  ·  GPU {gpu}"
        if self.paused:
            text += "  ·  Paused"
        return text

    def _draw_overlay(self, now: float) -> None:
        lines = [self._metrics_text()] if self.show_stats else []
        if now < self._status_until:
            lines.append(self._status)
        if not lines:
            return

        style = imgui.get_style()
        padding = style.window_padding
        width = max(imgui.calc_text_size(line).x for line in lines)
        line_height = imgui.get_text_line_height_with_spacing()
        right = imgui.get_io().display_size.x - padding.x
        left = max(padding.x, right - width - 2 * padding.x)
        top = padding.y
        bottom = top + 2 * padding.y + len(lines) * line_height - style.item_spacing.y
        overlay = imgui.get_foreground_draw_list()
        overlay.add_rect_filled(
            (left, top), (right, bottom), imgui.get_color_u32((0.06, 0.07, 0.08, 0.82)),
            rounding=padding.y,
        )
        overlay.push_clip_rect((left + padding.x, top), (right - padding.x, bottom), True)
        for row, line in enumerate(lines):
            overlay.add_text(
                (left + padding.x, top + padding.y + row * line_height),
                imgui.get_color_u32(imgui.Col_.text),
                line,
            )
        overlay.pop_clip_rect()

    def _run_frames(self, renderer: spy.ui.Context) -> None:
        target = None
        index = 0
        elapsed = average_dt = 0.0
        previous = perf_counter()
        ui_scale = None
        renderer_size = None
        style = imgui.get_style()
        spacing = {
            "window_padding": tuple(style.window_padding),
            "frame_padding": tuple(style.frame_padding),
            "item_spacing": tuple(style.item_spacing),
            "item_inner_spacing": tuple(style.item_inner_spacing),
        }

        while not self.window.should_close():
            self.window.process_events()
            if self.window.should_close():
                break
            if self._inspect_requested and target is not None:
                self._inspect_frame(target)
                previous = perf_counter()
            if self._pending_size is not None:
                self._resize(*self._pending_size)
                self._pending_size = None
            self._poll_gpu()
            image = (
                self.surface.acquire_next_image()
                if self.surface.config is not None
                else None
            )
            if image is None:
                sleep(0.016)
                previous = perf_counter()
                average_dt = 0.0
                continue

            now = perf_counter()
            dt, previous = now - previous, now
            average_dt = 0.9 * average_dt + 0.1 * dt if average_dt else dt
            self.fps = 1 / average_dt if average_dt > 0 else 0.0
            if ui_scale != self._ui_scale:
                style.font_scale_main = ui_scale = self._ui_scale
                for name, (x, y) in spacing.items():
                    setattr(style, name, (x * ui_scale, y * ui_scale))
            # SlangPy's surface and native renderer use the window coordinate size.
            imgui_backend.begin_frame(image.width, image.height, max(dt, 1e-6))
            self.draw_ui()
            self._draw_overlay(now)
            imgui.render()
            draw_data = imgui.get_draw_data()
            imgui_backend.sync_draw_data_textures(self.device, renderer, draw_data)

            encoder = self.device.create_command_encoder()
            size = image.width, image.height
            pool = None
            timestamp = self._next_timestamp
            previous_module = None
            if target is None or not self.paused:
                if target is None or size != (target.width, target.height):
                    target = self.device.create_texture(
                        width=size[0],
                        height=size[1],
                        format=spy.Format.rgba16_float,
                        usage=spy.TextureUsage.shader_resource
                        | spy.TextureUsage.unordered_access
                        | spy.TextureUsage.render_target
                        | spy.TextureUsage.copy_source
                        | spy.TextureUsage.copy_destination,
                        label="SlangToy frame",
                    )
                frame_dt = 0.0 if self.paused else dt
                elapsed += frame_dt
                previous_module = self._check_reload(now)
                pool = self._query_pool if len(self._timings) < _TIMING_SLOTS else None
                if pool is not None:
                    pool.reset(timestamp, 2)
                    encoder.write_timestamp(pool, timestamp)
                try:
                    self.render(Frame(target, encoder, index, elapsed, frame_dt))
                except (RuntimeError, ValueError, AttributeError) as error:
                    if previous_module is None:
                        raise
                    self.module, previous_module = previous_module, None
                    self._report_reload(error)
                    encoder = self.device.create_command_encoder()
                    if pool is not None:
                        encoder.write_timestamp(pool, timestamp)
                    self.render(Frame(target, encoder, index, elapsed, frame_dt))
                if pool is not None:
                    encoder.write_timestamp(pool, timestamp + 1)
                index += 1
            encoder.blit(image, target)
            if size != renderer_size:
                # SlangPy 0.43.1 reads the viewport dimensions from its native context.
                renderer.begin_frame(*size)
                renderer.end_frame(image, encoder)
                renderer_size = size
            imgui_backend.render_imgui_draw_data(renderer, draw_data, image, encoder)

            submit_id = self.device.submit_command_buffer(encoder.finish())
            if pool is not None:
                self._timings.append((timestamp, submit_id))
                self._next_timestamp = (timestamp + 2) % (2 * _TIMING_SLOTS)
            del image
            self.surface.present()
            if previous_module is not None:
                self._report_reload()
            self._save_screenshots(target)
