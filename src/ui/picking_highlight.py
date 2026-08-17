"""Point picking & selection highlight for the 3D tag map tab.

Mixin: methods operate on the ``TagMap3DTab`` instance (``self``). Covers 3D
point picking, node/cluster highlighting, and scatter plot updates. Moved here
from ``tag_map_3d_tab.py`` to reduce its size without changing behavior.
"""
import numpy as np


class PickingHighlightMixin:
    def pick_nearest_point(self, gl_view, click_pos, pick_threshold=20.0):
        """Vectorized 3D point picking using numpy matrix multiplication.

        Projects all points to screen space in one vectorized operation
        instead of iterating in a Python loop. ~100x faster at scale.

        Args:
            gl_view: The GLViewWidget instance (provides view/proj matrices).
            click_pos: QPoint of the click in widget coordinates.
            pick_threshold: Base pixel threshold for hit detection.

        Returns:
            int or None: Index of the nearest point within threshold, or None.
        """
        import numpy as np

        scatter = self.gl_scatter
        if scatter is None or scatter.pos is None or len(scatter.pos) == 0:
            return None

        positions = scatter.pos if isinstance(scatter.pos, np.ndarray) else np.array(scatter.pos)
        n = len(positions)
        if n == 0:
            return None

        # Build MVP matrix as numpy (4x4)
        view_matrix = gl_view.viewMatrix()
        proj_matrix = gl_view.currentProjection()
        mvp = proj_matrix * view_matrix

        # Extract 4x4 matrix from QMatrix4x4
        m = np.array([
            [mvp(0,0), mvp(0,1), mvp(0,2), mvp(0,3)],
            [mvp(1,0), mvp(1,1), mvp(1,2), mvp(1,3)],
            [mvp(2,0), mvp(2,1), mvp(2,2), mvp(2,3)],
            [mvp(3,0), mvp(3,1), mvp(3,2), mvp(3,3)],
        ])

        # Homogeneous coordinates: (n, 4) @ (4, 4).T -> (n, 4)
        ones = np.ones((n, 1))
        pos_h = np.hstack([positions, ones])  # (n, 4)
        clip = pos_h @ m.T  # (n, 4)

        # Perspective divide
        w = clip[:, 3]  # (n,)
        # Avoid division by zero
        w_safe = np.where(np.abs(w) < 1e-10, 1e-10, w)  # (n,)
        ndc = clip[:, :3] / w_safe[:, np.newaxis]  # (n, 3)

        # Filter: behind camera or outside NDC z range
        valid = (ndc[:, 2] >= 0) & (ndc[:, 2] <= 1) & (w > 0)
        if not np.any(valid):
            return None

        # Convert NDC to screen coordinates
        width = gl_view.width()
        height = gl_view.height()
        screen_x = (ndc[:, 0] * 0.5 + 0.5) * width
        screen_y = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * height

        # Distance from click to each point (vectorized)
        dx = screen_x - click_pos.x()
        dy = screen_y - click_pos.y()
        dist = np.sqrt(dx * dx + dy * dy)

        # Apply per-point size threshold if available
        if isinstance(scatter.size, np.ndarray) and len(scatter.size) == n:
            thresholds = pick_threshold + scatter.size
        else:
            base_size = scatter.size if isinstance(scatter.size, (int, float)) else 10
            thresholds = np.full(n, pick_threshold + base_size)

        # Mask: within threshold AND valid (in front of camera)
        dist[~valid] = np.inf
        dist[dist > thresholds] = np.inf

        closest_idx = int(np.argmin(dist))
        if dist[closest_idx] == np.inf:
            return None
        return closest_idx

    def on_node_clicked(self, scatter, ids):
        """Handle node click event to display file info.

        Args:
            scatter: The scatter plot item that was clicked
            ids: Object IDs that were clicked
        """
        if ids is None or len(ids) == 0:
            return

        # Get the first clicked ID
        node_index = ids[0]
        self.show_node_info(node_index)
    
    def _toggle_selection_highlight(self):
        """Toggle the visual highlight for selected node/cluster.

        Uses a separate highlight scatter item (only the selected points)
        so the blink timer just toggles visibility — zero GPU upload per tick.
        """
        if self.gl_scatter is None or self.node_list is None:
            return

        # Toggle visibility state
        self.selection_visible = not self.selection_visible

        # Just toggle the highlight item's visibility (no color re-upload)
        if hasattr(self, 'gl_highlight') and self.gl_highlight is not None:
            self.gl_highlight.setVisible(self.selection_visible)
    
    def _build_base_scatter(self):
        """Build and cache the base scatter (positions/sizes/base colors/cluster_ids).

        Returns the base colors_rgba array (no highlight/dim).
        """
        import numpy as np
        scene = self.scene_graph
        positions = scene.positions * float(self.spread_spin.value())
        node_size = self.min_size_spin.value() / 10.0
        sizes = np.full(len(scene.file_ids), node_size)
        colors = scene.colors.astype(np.float64) / 255.0
        alpha = self.transparency_spin.value()
        colors_rgba = np.column_stack([colors, alpha * np.ones(len(colors))])
        # Cache cluster_ids as numpy array (avoids O(n) Python loop per blink tick)
        self._base_cluster_ids = scene.cluster_ids.astype(np.int32)
        self._base_positions = positions
        self._base_sizes = sizes
        self._base_colors_rgba = colors_rgba
        # Re-spawn twinkle nodes if active (positions may have changed)
        if getattr(self, 'twinkle_active', False):
            self._spawn_twinkle_nodes()
        return positions, sizes, colors_rgba

    def _apply_highlight_colors(self, colors_rgba):
        """Update the existing scatter's colors in-place (no recreate)."""
        if self.gl_scatter is None:
            return
        self.gl_scatter.setData(color=colors_rgba)

    def _highlight_single_node(self, node_index):
        """Apply highlight for a single node using a separate highlight item.

        The base scatter gets dimmed once (persistent). A small highlight
        scatter shows the selected node. Blink just toggles visibility.
        """
        if not (0 <= node_index < len(self.node_list)):
            return
        import numpy as np

        if not hasattr(self, '_base_colors_rgba') or self._base_colors_rgba is None:
            self._build_base_scatter()

        # Dim base scatter (persistent, applied once)
        if self.dim_non_selected_checkbox.isChecked():
            colors_rgba = self._base_colors_rgba.copy()
            dim_alpha = self.dim_alpha_spin.value()
            mask = np.ones(len(colors_rgba), dtype=bool)
            mask[node_index] = False
            colors_rgba[mask, 3] = dim_alpha
            self._apply_highlight_colors(colors_rgba)
        else:
            self._apply_highlight_colors(self._base_colors_rgba)

        # Create/update highlight item (single point)
        self._update_highlight_item(
            positions=self._base_positions[node_index:node_index+1],
            color=[self.highlight_color[0], self.highlight_color[1], self.highlight_color[2], 1.0]
        )

    def _highlight_cluster(self, cluster_id):
        """Apply highlight for a cluster using a separate highlight item.

        The base scatter gets dimmed once (persistent). A small highlight
        scatter shows the selected cluster's points. Blink just toggles visibility.
        """
        import numpy as np

        if not hasattr(self, '_base_colors_rgba') or self._base_colors_rgba is None:
            self._build_base_scatter()

        # Dim base scatter (persistent, applied once)
        if self.dim_non_selected_checkbox.isChecked():
            colors_rgba = self._base_colors_rgba.copy()
            cluster_ids = self._base_cluster_ids
            colors_rgba[cluster_ids != cluster_id, 3] = self.dim_alpha_spin.value()
            self._apply_highlight_colors(colors_rgba)
        else:
            self._apply_highlight_colors(self._base_colors_rgba)

        # Create/update highlight item (cluster points only)
        cluster_ids = self._base_cluster_ids
        mask = cluster_ids == cluster_id
        self._update_highlight_item(
            positions=self._base_positions[mask],
            color=[self.highlight_color[0], self.highlight_color[1], self.highlight_color[2], 1.0]
        )

    def _update_highlight_item(self, positions, color):
        """Create or update the separate highlight scatter item.

        Only contains the selected points (not all 2M), so GPU upload
        is tiny. Blink timer just toggles visibility.
        """
        import numpy as np
        import pyqtgraph.opengl as gl

        if len(positions) == 0:
            self._remove_highlight_item()
            return

        # Remove old highlight item if exists
        self._remove_highlight_item()

        n = len(positions)
        colors = np.tile(np.array(color, dtype=np.float32), (n, 1))
        sizes = np.full(n, self.min_size_spin.value() / 10.0 * 1.5)  # Slightly larger

        self.gl_highlight = gl.GLScatterPlotItem(
            pos=positions,
            size=sizes,
            color=colors,
            pxMode=False
        )
        self.gl_highlight.setVisible(self.selection_visible)
        self.gl_view.addItem(self.gl_highlight)

    def _remove_highlight_item(self):
        """Remove the highlight scatter item if it exists."""
        if hasattr(self, 'gl_highlight') and self.gl_highlight is not None:
            try:
                self.gl_view.removeItem(self.gl_highlight)
            except (ValueError, KeyError):
                pass
            self.gl_highlight = None
    
    def _update_scatter_plot(self, positions, sizes, colors_rgba):
        """Update the scatter plot with new positions, sizes, and colors."""
        import pyqtgraph.opengl as gl
        
        # Remove old scatter (guard against already-removed item)
        if self.gl_scatter is not None:
            try:
                self.gl_view.removeItem(self.gl_scatter)
            except (ValueError, KeyError):
                pass  # Item already removed
        
        # Create new scatter
        self.gl_scatter = gl.GLScatterPlotItem(
            pos=positions,
            size=sizes,
            color=colors_rgba,
            pxMode=False
        )
        try:
            self.gl_scatter.sigClicked.connect(self.on_node_clicked)
        except (AttributeError, TypeError):
            pass
        self.gl_view.addItem(self.gl_scatter)
    
    def _reapply_selection_style(self):
        """Re-apply the current selection highlight/dim style to the scatter."""
        if self.gl_scatter is None or self.node_list is None:
            return
        if self.selected_node_index is not None:
            self._highlight_single_node(self.selected_node_index)
        elif self.selected_cluster_id is not None:
            self._highlight_cluster(self.selected_cluster_id)
