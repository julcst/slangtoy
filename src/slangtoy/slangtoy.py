# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""A SlangPy window for experimenting with interactive shaders."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter, sleep

import slangpy as spy

_TIMING_SLOTS = 4
_LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Frame:
    """Render inputs for one frame. Time and dt are in seconds.

    Time follows pause state. Index and dt track rendered frames while paused.
    Record commands during render and let the window handle submission.
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
    """Override setup() to build widgets on ui.screen and render() to draw.

    Pass a shader path to expose a hot-reloadable SlangPy module as self.module.
    Use the frame's encoder for all drawing to include it in GPU timings.
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
        self.ui = spy.ui.Context(self.device)
        self.paused = False
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
        self._pending_size: tuple[int, int] | None = None
        self._status = ""
        self._status_until = 0.0

    def setup(self) -> None:
        """Create custom resources and UI widgets once, before the first frame."""

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
        """Queue a PNG of the next rendered frame before the UI is drawn."""
        if path is None:
            path = Path(f"screenshot-{datetime.now():%Y%m%d-%H%M%S-%f}.png")
        path = Path(path)
        self._screenshots.append(path)
        return path

    def _keyboard_event(self, event: spy.KeyboardEvent) -> None:
        if self.ui.handle_keyboard_event(event):
            return
        if event.type == spy.KeyboardEventType.key_press:
            match event.key:
                case spy.KeyCode.escape:
                    self.window.close()
                case spy.KeyCode.space:
                    self.paused = not self.paused
                case spy.KeyCode.f2:
                    self.screenshot()
                case spy.KeyCode.f3:
                    self.show_stats = not self.show_stats
                case spy.KeyCode.f5 if self._watch:
                    self._reload_requested = True
                case _:
                    self.on_key(event)
        else:
            self.on_key(event)

    def _mouse_event(self, event: spy.MouseEvent) -> None:
        if not self.ui.handle_mouse_event(event):
            self.on_mouse(event)

    def _shader_stamps(self) -> tuple[int | None, ...]:
        stamps = []
        for path in self._watch:
            try:
                stamps.append(path.stat().st_mtime_ns)
            except OSError:
                stamps.append(None)
        return tuple(stamps)

    def _reload_failed(self, error: Exception) -> None:
        self._notify("Shader reload failed. See terminal.", error)

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
            self._reload_failed(error)
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

    def run(self) -> None:
        """Open the window loop on the main thread and run until it closes."""
        self.window.on_resize = self._on_resize
        self.window.on_keyboard_event = self._keyboard_event
        self.window.on_mouse_event = self._mouse_event
        try:
            self._resize(self.window.width, self.window.height)
            self.setup()
            self._run_frames()
        finally:
            self.window.on_resize = None
            self.window.on_keyboard_event = None
            self.window.on_mouse_event = None
            self.device.wait()
            self.surface.unconfigure()

    def _metrics_text(self) -> str:
        gpu = "n/a" if self._query_pool is None else "…"
        if self.gpu_ms is not None:
            gpu = f"{self.gpu_ms:.2f} ms"
        text = f"{self.fps:.1f} FPS  ·  GPU {gpu}"
        if self.paused:
            text += "  ·  Paused"
        return text

    def _run_frames(self) -> None:
        target = None
        panel = spy.ui.Window(
            self.ui.screen, "Stats", position=spy.float2(10, 10), size=spy.float2(238, 43)
        )
        metrics = spy.ui.Text(panel)
        notification = spy.ui.Text(panel)
        notification.visible = False
        index = 0
        elapsed = average_dt = 0.0
        previous = perf_counter()

        while not self.window.should_close():
            self.window.process_events()
            if self.window.should_close():
                break
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
            previous_module = self._check_reload(now)
            average_dt = 0.9 * average_dt + 0.1 * dt if average_dt else dt
            self.fps = 1 / average_dt if average_dt > 0 else 0.0
            if not self.paused:
                elapsed += dt
            size = image.width, image.height
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

            pool = self._query_pool if len(self._timings) < _TIMING_SLOTS else None
            timestamp = self._next_timestamp
            if pool is not None:
                pool.reset(timestamp, 2)

            while True:
                encoder = self.device.create_command_encoder()
                if pool is not None:
                    encoder.write_timestamp(pool, timestamp)
                try:
                    self.render(Frame(target, encoder, index, elapsed, dt))
                except (RuntimeError, ValueError, AttributeError) as error:
                    if previous_module is None:
                        raise
                    self.module, previous_module = previous_module, None
                    self._reload_failed(error)
                else:
                    if previous_module is not None:
                        self._notify("Shader reloaded")
                    break

            encoder.blit(image, target)
            stats_visible = self.show_stats
            panel.visible = stats_visible
            if stats_visible:
                metrics.text = self._metrics_text()
                show_notification = now < self._status_until
                if show_notification != notification.visible:
                    panel.size = (
                        spy.float2(320, 62) if show_notification else spy.float2(238, 43)
                    )
                    notification.visible = show_notification
                notification.text = self._status
            self.ui.begin_frame(*size, window=self.window)
            if stats_visible and not panel.visible:
                self.show_stats = False
            self.ui.end_frame(image, encoder)

            if pool is not None:
                encoder.write_timestamp(pool, timestamp + 1)
            submit_id = self.device.submit_command_buffer(encoder.finish())
            if pool is not None:
                self._timings.append((timestamp, submit_id))
                self._next_timestamp = (timestamp + 2) % (2 * _TIMING_SLOTS)
            del image
            self.surface.present()
            self._save_screenshots(target)
            index += 1
