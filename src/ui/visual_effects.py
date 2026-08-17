"""Visual effects for the 3D tag map tab.

Mixin: methods operate on the ``TagMap3DTab`` instance (``self``). Covers dim,
twinkle, and wobble visual effects. Moved here from ``tag_map_3d_tab.py`` to
reduce its size without changing behavior.
"""
import numpy as np


class VisualEffectsMixin:
    def _on_dim_toggle_changed(self):
        """Handle dim non-selected toggle changes."""
        self._reapply_selection_style()
        # Also dim/undim the cohort labels of non-selected cohorts
        self._update_cohort_labels()

    def _on_dim_alpha_changed(self):
        """Handle dim alpha parameter changes."""
        self._reapply_selection_style()
        # Also update the cohort label dim alpha
        self._update_cohort_labels()

    def _pick_highlight_color(self):
        """Open color picker for the highlight/blink color."""
        from PySide6.QtWidgets import QColorDialog
        from PySide6.QtGui import QColor
        current = QColor(int(self.highlight_color[0]*255), int(self.highlight_color[1]*255), int(self.highlight_color[2]*255))
        color = QColorDialog.getColor(current, self, "Highlight Color")
        if color.isValid():
            self.highlight_color = (color.red()/255.0, color.green()/255.0, color.blue()/255.0)
            r, g, b = int(color.red()), int(color.green()), int(color.blue())
            self.highlight_color_btn.setStyleSheet(f"background-color: rgb({r},{g},{b}); color: black; font-weight: bold;")
            # Re-apply if a selection is active
            self._reapply_selection_style()

    # ─── Star Twinkle Effect ───────────────────────────────────────────────

    def _on_twinkle_toggle(self, state):
        """Start or stop the star twinkle animation."""
        # Guard: state attrs may not exist yet during early load_settings()
        if not hasattr(self, 'twinkle_timer'):
            return
        self.twinkle_active = bool(state)
        if self.twinkle_active:
            if not hasattr(self, '_base_positions') or self._base_positions is None:
                # No scene loaded yet; disable the checkbox
                self.twinkle_checkbox.setChecked(False)
                return
            self._spawn_twinkle_nodes()
            self.twinkle_timer.start(33)  # ~30 fps
        else:
            self.twinkle_timer.stop()
            self._remove_twinkle_item()

    def _on_twinkle_param_changed(self):
        """Restart twinkle with new parameters if active."""
        if not hasattr(self, 'twinkle_active'):
            return
        if self.twinkle_active and hasattr(self, '_base_positions') and self._base_positions is not None:
            self._spawn_twinkle_nodes()

    def _spawn_twinkle_nodes(self):
        """Pick a random set of nodes and assign them lifespans/phases."""
        n_total = len(self._base_positions)
        if n_total == 0:
            return
        count = min(self.twinkle_count_spin.value(), n_total)
        now = time.perf_counter()

        self.twinkle_indices = np.random.choice(n_total, size=count, replace=False).astype(np.int32)
        lifespan_min = self.twinkle_lifespan_min_spin.value()
        lifespan_max = max(self.twinkle_lifespan_max_spin.value(), lifespan_min)
        # All nodes are born now; the random lifespan distribution naturally staggers
        # their deaths over time (short-lived ones get replaced first, long-lived later).
        self.twinkle_birth = np.full(count, now)
        self.twinkle_lifespan = np.random.uniform(lifespan_min, lifespan_max, size=count)
        self.twinkle_phase = np.random.uniform(0, 2 * np.pi, size=count)

        # Create the twinkle scatter item (positions are static, colors update per frame)
        self._remove_twinkle_item()
        import pyqtgraph.opengl as gl
        positions = self._base_positions[self.twinkle_indices]
        node_size = self.min_size_spin.value() / 10.0
        sizes = np.full(count, node_size * 1.2)
        # Initial colors: base colors of those nodes
        base_colors = self._base_colors_rgba[self.twinkle_indices] if hasattr(self, '_base_colors_rgba') else None
        if base_colors is None:
            base_colors = np.tile([0.5, 0.5, 0.5, 1.0], (count, 1))
        self.gl_twinkle = gl.GLScatterPlotItem(
            pos=positions, size=sizes, color=base_colors, pxMode=False
        )
        self.gl_view.addItem(self.gl_twinkle)

    def _update_twinkle(self):
        """Per-frame twinkle update: compute brightness, replace dead nodes."""
        if not self.twinkle_active or self.gl_twinkle is None:
            return
        now = time.perf_counter()
        freq = self.twinkle_freq_spin.value()
        brightness_mult = self.twinkle_brightness_spin.value()

        # Compute per-node blink intensity: 0.5 + 0.5 * sin(2π * freq * t + phase)
        age = now - self.twinkle_birth
        intensity = 0.5 + 0.5 * np.sin(2 * np.pi * freq * age + self.twinkle_phase)

        # Get base colors for active nodes
        if not hasattr(self, '_base_colors_rgba') or self._base_colors_rgba is None:
            return
        base = self._base_colors_rgba[self.twinkle_indices]  # (n, 4)

        # Apply brightness: scale RGB by (1 + intensity * brightness_mult), keep alpha high
        colors = base.copy()
        rgb_scale = 1.0 + intensity * brightness_mult
        colors[:, 0] = np.clip(base[:, 0] * rgb_scale, 0, 1)
        colors[:, 1] = np.clip(base[:, 1] * rgb_scale, 0, 1)
        colors[:, 2] = np.clip(base[:, 2] * rgb_scale, 0, 1)
        colors[:, 3] = np.clip(0.5 + intensity * 0.5, 0, 1)

        self.gl_twinkle.setData(color=colors)

        # Replace dead nodes (age > lifespan) with new random ones
        dead_mask = age > self.twinkle_lifespan
        if np.any(dead_mask):
            n_dead = int(np.sum(dead_mask))
            n_total = len(self._base_positions)
            # Pick new random indices (avoid current active set for variety)
            new_indices = np.random.choice(n_total, size=n_dead, replace=False).astype(np.int32)
            lifespan_min = self.twinkle_lifespan_min_spin.value()
            lifespan_max = max(self.twinkle_lifespan_max_spin.value(), lifespan_min)

            # Update arrays in place
            self.twinkle_indices[dead_mask] = new_indices
            self.twinkle_birth[dead_mask] = now
            self.twinkle_lifespan[dead_mask] = np.random.uniform(lifespan_min, lifespan_max, size=n_dead)
            self.twinkle_phase[dead_mask] = np.random.uniform(0, 2 * np.pi, size=n_dead)

            # Update positions in the scatter item (only changed points)
            new_positions = self._base_positions[new_indices]
            # Rebuild full position array for setData (cheap: only `count` points)
            all_positions = self._base_positions[self.twinkle_indices]
            self.gl_twinkle.setData(pos=all_positions)

    def _remove_twinkle_item(self):
        """Remove the twinkle scatter item if it exists."""
        if self.gl_twinkle is not None:
            try:
                self.gl_view.removeItem(self.gl_twinkle)
            except (ValueError, KeyError):
                pass
            self.gl_twinkle = None

    def _on_wobble_toggle(self, state):
        """Handle wobble enable/disable toggle."""
        from PySide6.QtCore import Qt
        if state == Qt.CheckState.Checked.value:
            # Capture the current camera angles as the base for oscillation/spin.
            self._capture_wobble_base()
            self.wobble_timer.start()
            print("Camera wobble enabled")
        else:
            self.wobble_timer.stop()
            print("Camera wobble disabled")

    def _on_wobble_continuous_toggle(self, state):
        """Handle continuous-spin toggle. Re-base so there is no camera jump."""
        if hasattr(self, 'gl_view') and self.gl_view is not None:
            self._capture_wobble_base()
        # Reset the spin accumulator so continuous mode starts from current azimuth.
        self.wobble_spin_azim = 0.0

    def _capture_wobble_base(self):
        """Store the current camera angles as the wobble/spin base."""
        if hasattr(self, 'gl_view') and self.gl_view is not None:
            self.wobble_base_azim = self.gl_view.opts.get('azimuth', 45)
            self.wobble_base_elev = self.gl_view.opts.get('elevation', 30)

    def _update_wobble(self):
        """Update camera position for wobble (oscillate) or continuous spin."""
        import numpy as np
        if not hasattr(self, 'gl_view') or self.gl_view is None:
            return

        # Pause while the user is dragging so manual orbit isn't fought.
        # Time is NOT advanced here, so the animation resumes smoothly on release.
        if getattr(self, 'wobble_user_interacting', False):
            return

        # Increment time
        dt = 0.016  # ~60fps timestep
        self.wobble_time += dt

        # Get wobble parameters
        speed = self.wobble_speed_spin.value()
        azim_range = self.wobble_azim_range_spin.value()
        elev_range = self.wobble_elev_range_spin.value()

        t = self.wobble_time * speed

        if self.wobble_continuous_checkbox.isChecked():
            # Continuous spin: azimuth rotates in one direction (wraps at 360),
            # elevation bobs gently so the camera never flips over the top.
            # The Azim/Elev Range values act as degrees-per-second.
            self.wobble_spin_azim += azim_range * dt * speed
            new_azimuth = (self.wobble_base_azim + self.wobble_spin_azim) % 360.0
            new_elevation = self.wobble_base_elev + np.sin(t * 0.4) * (elev_range * 0.5)
        else:
            # Classic sine wobble: oscillate around the base angles for parallax depth.
            new_azimuth = self.wobble_base_azim + np.sin(t * 0.5) * azim_range
            new_elevation = self.wobble_base_elev + np.sin(t * 0.7) * elev_range

        # Update camera position
        self.gl_view.opts['elevation'] = new_elevation
        self.gl_view.opts['azimuth'] = new_azimuth
        self.gl_view.update()
