"""Cohort label logic for the 3D tag map tab.

Mixin: methods operate on the ``TagMap3DTab`` instance (``self``). Covers all
cohort-label rendering, smart labels, color picking, and blink animation. Moved
here from ``tag_map_3d_tab.py`` to reduce its size without changing behavior.
"""
import numpy as np

from PySide6.QtCore import Qt

from src.ui.styles import BLUE_60
from src.ui.gl_text_items import get_multiline_text_item_class as _get_multiline_text_item_class


class CohortLabelsMixin:
    def _on_cohort_threshold_changed(self, value):
        """Update cohort labels when threshold changes."""
        self._update_cohort_labels()

    def _on_show_cohort_labels_toggled(self, state):
        """Toggle cohort labels in the 3D view."""
        self._update_cohort_labels()
        # Start/stop the label blink timer
        if hasattr(self, 'cohort_label_blink_timer'):
            if self.show_cohort_labels_checkbox.isChecked():
                self.cohort_label_blink_visible = True
                self.cohort_label_blink_timer.start(500)  # ~2Hz blink
            else:
                self.cohort_label_blink_timer.stop()

    def _on_label_fade_toggled(self, state):
        """Toggle whether dropping labels fade out on selection switch."""
        self.label_fade_enabled = bool(state)

    def _toggle_cohort_label_blink(self):
        """Blink the selected cohort's label between the two label colors."""
        self.cohort_label_blink_visible = not self.cohort_label_blink_visible
        # Determine the selected cohort's cluster_id
        selected_cid = self.selected_cluster_id
        if selected_cid is None and self.selected_node_index is not None:
            scene = getattr(self, 'scene_graph', None)
            if scene is not None and 0 <= self.selected_node_index < len(scene.file_ids):
                selected_cid = int(scene.cluster_ids[self.selected_node_index])
        # Swap only the selected cohort's label color; keep all others visible.
        # The custom paint() reads self.color (a QColor), so set that and
        # request a repaint via update().
        from PySide6.QtGui import QColor
        rgba = self._get_selected_label_rgba()
        for cid, item in self.cohort_label_map.items():
            try:
                if cid == selected_cid:
                    item.color = QColor(rgba[0], rgba[1], rgba[2], rgba[3])
                    item.update()
                else:
                    item.setVisible(True)
            except Exception:
                pass

    def _on_cohort_label_mode_changed(self, value):
        """Update cohort labels when the label mode changes."""
        self._update_cohort_labels()

    def _on_cohort_label_n_changed(self, value):
        """Update cohort labels when the N parameter changes."""
        self._update_cohort_labels()

    def _on_cohort_label_size_changed(self, value):
        """Update cohort label size."""
        self._update_cohort_labels()

    def _on_cohort_label_max_tags_changed(self, value):
        """Update cohort labels when the max-tags-per-label parameter changes."""
        self._update_cohort_labels()

    def _on_smart_label_mode_changed(self, value):
        """Update cohort labels when the smart label mode changes."""
        self._update_cohort_labels()

    def _on_dynamic_label_size_toggled(self, state):
        """Toggle dynamic label sizing."""
        self._update_cohort_labels()

    def _pick_cohort_label_color(self):
        """Open color picker for cohort labels."""
        from PySide6.QtWidgets import QColorDialog
        from PySide6.QtGui import QColor
        color = QColorDialog.getColor(
            QColor(*self._cohort_label_color),
            self,
            "Select Cohort Label Color",
        )
        if color.isValid():
            self._cohort_label_color = (color.red(), color.green(), color.blue())
            self._update_label_color_button()
            self._update_cohort_labels()

    def _update_label_color_button(self):
        """Update the label color button background."""
        r, g, b = self._cohort_label_color
        self.cohort_label_color_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: rgb({r}, {g}, {b});
                border: 1px solid {BLUE_60};
                border-radius: 3px;
            }}
        """)

    def _pick_cohort_label_color2(self):
        """Open color picker for the selected cohort's blink (secondary) label color."""
        from PySide6.QtWidgets import QColorDialog
        from PySide6.QtGui import QColor
        color = QColorDialog.getColor(
            QColor(*self._cohort_label_color2),
            self,
            "Select Blink Label Color",
        )
        if color.isValid():
            self._cohort_label_color2 = (color.red(), color.green(), color.blue())
            self._update_label_color_button2()
            self._update_cohort_labels()

    def _update_label_color_button2(self):
        """Update the blink (secondary) label color button background."""
        r, g, b = self._cohort_label_color2
        self.cohort_label_color2_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: rgb({r}, {g}, {b});
                border: 1px solid {BLUE_60};
                border-radius: 3px;
            }}
        """)

    def _pick_cohort_label_outline_color(self):
        """Open color picker for the cohort label outline."""
        from PySide6.QtWidgets import QColorDialog
        from PySide6.QtGui import QColor
        color = QColorDialog.getColor(
            QColor(*self._cohort_label_outline_color),
            self,
            "Select Label Outline Color",
        )
        if color.isValid():
            self._cohort_label_outline_color = (color.red(), color.green(), color.blue())
            self._update_outline_color_button()
            self._update_cohort_labels()

    def _update_outline_color_button(self):
        """Update the outline color button background."""
        r, g, b = self._cohort_label_outline_color
        self.cohort_label_outline_color_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: rgb({r}, {g}, {b});
                border: 1px solid {BLUE_60};
                border-radius: 3px;
            }}
        """)

    def _on_cohort_label_outline_width_changed(self, value):
        """Update label outline width and re-render labels."""
        self._update_cohort_labels()

    def _get_selected_label_rgba(self):
        """Return the RGBA tuple for the selected cohort's label based on blink state.

        The selected cohort's label blinks between the primary label color
        (when blink_visible) and the secondary blink color (when not).
        """
        if self.cohort_label_blink_visible:
            r, g, b = self._cohort_label_color
        else:
            r, g, b = self._cohort_label_color2
        return (r, g, b, 255)

    def _get_query_shared_tags(self):
        """Get tags that are AND-shared across all files (from the query).

        Returns:
            set: Tags that appear as plain (non-negative, non-OR) query terms.
        """
        query = self.query_edit.text().strip()
        shared = set()
        if not query:
            return shared
        for part in self._split_query_preserving_brackets(query):
            part = part.strip()
            if not part:
                continue
            # Skip OR brackets and negative tags
            if part.startswith('[') and part.endswith(']'):
                continue
            if part.startswith('-'):
                continue
            shared.add(part)
        return shared

    def _compute_cohort_label_text(self, cluster_id, nodes):
        """Compute the dominant-tag label text for a cohort.

        Filters out tags that are AND-shared across all files (from the query),
        since those tags are present in every cohort and provide no distinction.
        Tags are newline-separated (no percentages) to keep labels short and
        readable in the 3D view.
        """
        from collections import Counter
        count = len(nodes)
        scene = self.scene_graph
        tag_data = self.tag_data or {}
        tag_counter = Counter()
        for i in nodes:
            tags = tag_data.get(scene.file_ids[i], [])
            if self.tag_interner:
                tag_counter.update(self.tag_interner.strings_to_list(tags))
            else:
                tag_counter.update(tags)

        shared_tags = self._get_query_shared_tags()
        threshold = self.cohort_threshold_spin.value()
        max_tags = self.cohort_label_max_tags_spin.value()

        dominant_tags = []
        for tag, tag_count in tag_counter.most_common():
            if tag in shared_tags:
                continue  # Skip AND-shared tags (present in all cohorts)
            percentage = tag_count / count
            if percentage >= threshold:
                dominant_tags.append(tag)

        # If no tags meet the threshold, fall back to top tags (still filtered)
        if not dominant_tags:
            for tag, tag_count in tag_counter.most_common(max_tags):
                if tag in shared_tags:
                    continue
                dominant_tags.append(tag)

        return dominant_tags[:max_tags]

    def _get_cohort_dominant_tags_full(self, nodes):
        """Get the FULL ranked dominant-tag list for a cohort (not truncated).

        Skips AND-shared query tags but does NOT apply the cohort threshold.
        Returns the complete ranked list so smart-labels resolution can pick
        next-in-line tags when the top ones are taken.

        Returns:
            list: All non-shared tags in dominance order (tag, count) tuples.
        """
        from collections import Counter
        scene = self.scene_graph
        tag_data = self.tag_data or {}
        tag_counter = Counter()
        for i in nodes:
            tags = tag_data.get(scene.file_ids[i], [])
            if self.tag_interner:
                tag_counter.update(self.tag_interner.strings_to_list(tags))
            else:
                tag_counter.update(tags)

        shared_tags = self._get_query_shared_tags()

        ranked = []
        for tag, tag_count in tag_counter.most_common():
            if tag in shared_tags:
                continue
            ranked.append((tag, tag_count))

        return ranked

    def _apply_smart_labels(self, cluster_nodes):
        """Resolve duplicate labels across cohorts (smart labels).

        Modes:
        - "All Unique" / "Overlap" (proximity-based):
            For each cohort, find the top 5 closest cohorts by centroid
            distance. If a cohort's label tags collide with a closer-or-
            equally-large neighbor's labels, the LARGER cohort keeps its
            tags and the smaller cohort substitutes the next-in-line
            dominant tags.
            - All Unique: replace ALL colliding tags with free ones.
            - Overlap: keep the FIRST tag, replace the rest with free ones.
        - "Absolute Unique" (global):
            No tag may appear in more than one cohort label. Cohorts are
            processed biggest-first; each greedily reserves its top
            dominant tags, skipping already-taken ones. May run out of
            tags if there are many cohorts (acceptable).

        Args:
            cluster_nodes: dict of cluster_id -> list of nodes (already
                filtered by label mode).

        Returns:
            dict: cluster_id -> list of tag strings (final labels).
        """
        import numpy as np

        max_tags = self.cohort_label_max_tags_spin.value()
        mode = self.smart_label_mode_combo.currentText()

        # Precompute per-cohort data: centroid, size, full ranked tag list
        scene = self.scene_graph
        cohort_info = {}
        for cid, idx in cluster_nodes.items():
            centroid = scene.positions[idx].mean(axis=0)
            ranked = self._get_cohort_dominant_tags_full(idx)
            cohort_info[cid] = {
                "centroid": centroid,
                "size": len(idx),
                "ranked": ranked,
            }

        if mode == "Absolute Unique":
            # Global greedy reservation, biggest cohort first
            taken = set()
            result = {}
            for cid in sorted(cohort_info, key=lambda c: cohort_info[c]["size"], reverse=True):
                info = cohort_info[cid]
                chosen = []
                for tag, _count in info["ranked"]:
                    if tag in taken:
                        continue
                    chosen.append(tag)
                    taken.add(tag)
                    if len(chosen) >= max_tags:
                        break
                result[cid] = chosen
            return result

        # Proximity-based modes (All Unique / Overlap)
        cids = list(cohort_info.keys())
        n = len(cids)
        if n < 2:
            return {cid: [t for t, _ in info["ranked"]][:max_tags]
                    for cid, info in cohort_info.items()}

        # Centroid matrix for distance computation
        centroids = np.array([cohort_info[c]["centroid"] for c in cids])
        # Pairwise distance matrix (n x n)
        diff = centroids[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        dist_matrix = np.sqrt((diff ** 2).sum(axis=2))

        # For each cohort, find top 5 closest other cohorts
        closest = {}
        for i, cid in enumerate(cids):
            row = dist_matrix[i].copy()
            row[i] = np.inf
            k = min(5, n - 1)
            idx = np.argsort(row)[:k]
            closest[cid] = [cids[j] for j in idx]

        # Initial labels: top max_tags dominant tags
        labels = {
            cid: [t for t, _ in info["ranked"]][:max_tags]
            for cid, info in cohort_info.items()
        }

        # Resolve collisions. Process cohorts smallest-first so the larger
        # cohort always keeps its tags (the smaller one yields).
        for cid in sorted(cids, key=lambda c: cohort_info[c]["size"]):
            my_neighbors = closest[cid]
            # Collect tags held by neighbors that are LARGER than us
            # (we yield to larger neighbors; equal size -> lower cid yields)
            neighbor_tags = set()
            for nb in my_neighbors:
                nb_info = cohort_info[nb]
                if nb_info["size"] > cohort_info[cid]["size"]:
                    neighbor_tags.update(labels[nb])
                elif nb_info["size"] == cohort_info[cid]["size"] and nb < cid:
                    neighbor_tags.update(labels[nb])
            if not neighbor_tags:
                continue

            my_tags = labels[cid]
            colliding = [t for t in my_tags if t in neighbor_tags]
            if not colliding:
                continue

            if mode == "Overlap" and len(my_tags) >= 2:
                # Keep the first tag, replace the rest that collide
                keep = [my_tags[0]]
                to_replace = [t for t in my_tags[1:] if t in neighbor_tags]
                if not to_replace:
                    continue
                # Fill with next-in-line free tags
                for tag, _count in cohort_info[cid]["ranked"]:
                    if len(keep) >= max_tags:
                        break
                    if tag in keep or tag in neighbor_tags:
                        continue
                    keep.append(tag)
                labels[cid] = keep[:max_tags]
            else:
                # All Unique: replace ALL colliding tags with free ones
                kept = [t for t in my_tags if t not in neighbor_tags]
                for tag, _count in cohort_info[cid]["ranked"]:
                    if len(kept) >= max_tags:
                        break
                    if tag in kept or tag in neighbor_tags:
                        continue
                    kept.append(tag)
                labels[cid] = kept[:max_tags]

        return labels

    def _capture_dropping_labels(self, target_cid=None):
        """Start fading out labels that are about to drop due to a selection switch.

        Must be called BEFORE the old labels are removed (i.e. before
        clear_selection / _remove_cohort_labels). Only active when:
          - "Fade label transitions" is enabled, and
          - label mode is "Selected & N neighbors", and
          - there's a current selection with existing labels.

        It computes which cohorts will remain after the switch (selected + its N
        nearest) and fades out every currently-shown label that won't be in that
        set. Kept labels are rebuilt fresh by _update_cohort_labels as usual.

        Args:
            target_cid: the cohort being selected next. When provided, it's used
                to compute the keep-set (the selection state may not have been
                updated yet at call time). If None, resolved from current state.
        """
        if not getattr(self, 'label_fade_enabled', True):
            return
        try:
            import numpy as np
            mode = self.cohort_label_mode_combo.currentText()
            if mode != "Selected & N neighbors":
                return
            old_map = getattr(self, 'cohort_label_map', {}) or {}
            if not old_map:
                return
            scene = getattr(self, 'scene_graph', None)
            if scene is None or not hasattr(self, 'node_list') or not self.node_list:
                return

            # Resolve the (new) selected cohort. Prefer the explicit target so this
            # works when called before the selection state has been updated.
            selected_cid = target_cid
            if selected_cid is None:
                selected_cid = self.selected_cluster_id
                if selected_cid is None and self.selected_node_index is not None:
                    if 0 <= self.selected_node_index < len(scene.file_ids):
                        selected_cid = int(scene.cluster_ids[self.selected_node_index])
            if selected_cid is None or selected_cid == -1:
                return

            # Build the set of cohorts that will remain after the switch.
            cids_arr = scene.cluster_ids
            n = self.cohort_label_n_spin.value()
            cluster_nodes = {}
            for cid in np.unique(cids_arr):
                if int(cid) == -1:
                    continue
                cluster_nodes[int(cid)] = np.where(cids_arr == cid)[0]

            keep = set()
            if selected_cid in cluster_nodes:
                sel_centroid = scene.positions[cluster_nodes[selected_cid]].mean(axis=0)
                others = {cid: idx for cid, idx in cluster_nodes.items() if cid != selected_cid}

                def _centroid(idx):
                    return scene.positions[idx].mean(axis=0)

                ranked = sorted(
                    others.items(),
                    key=lambda kv: float(np.linalg.norm(_centroid(kv[1]) - sel_centroid))
                )
                keep.add(selected_cid)
                for cid, _nodes in ranked[:n]:
                    keep.add(cid)

            # Fade out every currently-shown label that won't remain.
            fading = []
            for cid, item in old_map.items():
                if cid in keep:
                    continue
                try:
                    c = item.color
                    base_rgba = (int(c.red()), int(c.green()), int(c.blue()), max(1, int(c.alpha())))
                    fading.append([item, base_rgba])
                except Exception:
                    pass
            if fading:
                self._start_label_fade(fading)
        except Exception as e:
            print(f"Error capturing dropping labels: {e}")

    def _update_cohort_labels(self):
        """Render dominant-tag labels centered on each cohort in the 3D view."""
        try:
            import pyqtgraph.opengl as gl
            import numpy as np

            if not hasattr(self, 'gl_view') or self.gl_view is None:
                return

            # NOTE: dropping-label capture for selection switches happens in
            # show_cluster_info() (before clear_selection removes the old labels),
            # NOT here — calling it here would double-capture and restart fades at
            # a reduced alpha. This method only rebuilds the kept labels; items
            # already mid-fade are preserved by _remove_cohort_labels(keep_ids).

            if not self.show_cohort_labels_checkbox.isChecked():
                self._remove_cohort_labels()
                return
            if not hasattr(self, 'node_list') or not self.node_list:
                self._remove_cohort_labels()
                return

            # Group member indices by cluster (vectorized; skip noise)
            scene = self.scene_graph
            cids_arr = scene.cluster_ids
            cluster_nodes = {}
            for cid in np.unique(cids_arr):
                if int(cid) == -1:
                    continue  # Skip noise
                cluster_nodes[int(cid)] = np.where(cids_arr == cid)[0]

            if not cluster_nodes:
                self._remove_cohort_labels()
                return

            # Determine label color
            r, g, b = self._cohort_label_color

            # Compute max cohort size for dynamic scaling
            max_count = max(len(nodes) for nodes in cluster_nodes.values()) if cluster_nodes else 1
            base_size = self.cohort_label_size_spin.value()

            # Resolve the currently selected cohort (a single-node selection
            # counts as its cohort). Needed by both "Selected Only" mode and
            # the always-include-selected fallback below.
            selected_cid = self.selected_cluster_id
            if selected_cid is None and self.selected_node_index is not None:
                if 0 <= self.selected_node_index < len(scene.file_ids):
                    selected_cid = int(scene.cluster_ids[self.selected_node_index])

            # Apply label mode filter to avoid overlap noise
            mode = self.cohort_label_mode_combo.currentText()
            n = self.cohort_label_n_spin.value()
            if mode == "Selected Only":
                # Show only the selected cohort's label (lightest). No-op if none selected.
                if selected_cid is not None:
                    cluster_nodes = {cid: nodes for cid, nodes in cluster_nodes.items() if cid == selected_cid}
                else:
                    cluster_nodes = {}
            elif mode == "Selected & N neighbors":
                # Selected cohort + its N nearest neighbors (by centroid distance).
                if selected_cid is not None and selected_cid in cluster_nodes:
                    sel_centroid = scene.positions[cluster_nodes[selected_cid]].mean(axis=0)
                    others = {cid: idx for cid, idx in cluster_nodes.items() if cid != selected_cid}
                    # Rank other cohorts by centroid distance to the selection
                    def _centroid(idx):
                        return scene.positions[idx].mean(axis=0)
                    ranked = sorted(
                        others.items(),
                        key=lambda kv: float(np.linalg.norm(_centroid(kv[1]) - sel_centroid))
                    )
                    keep = {selected_cid: cluster_nodes[selected_cid]}
                    for cid, nodes in ranked[:n]:
                        keep[cid] = nodes
                    cluster_nodes = keep
                else:
                    # No valid selection -> nothing to anchor neighbors to.
                    cluster_nodes = {}
            elif mode == "Top N largest":
                # Sort cohorts by size descending, keep top N
                sorted_cohorts = sorted(cluster_nodes.items(), key=lambda kv: len(kv[1]), reverse=True)
                cluster_nodes = dict(sorted_cohorts[:n])
            elif mode == "Above size threshold":
                # Keep only cohorts with at least N files
                cluster_nodes = {cid: nodes for cid, nodes in cluster_nodes.items() if len(nodes) >= n}
            # "All cohorts" -> no filtering

            # Always include the selected cohort so it gets a label even when
            # the label mode filter excluded it (e.g. outside top N or below
            # the size threshold).
            if selected_cid is not None and selected_cid not in cluster_nodes:
                sel_idx = np.where(scene.cluster_ids == selected_cid)[0]
                if len(sel_idx) > 0:
                    cluster_nodes[selected_cid] = sel_idx

            # Remove existing (non-fading) labels; kept ones are rebuilt fresh below.
            # Labels that are dropping due to a selection switch were already captured
            # and started fading by _capture_dropping_labels() before the clear, so
            # they're excluded here via keep_ids and removed by their fade timer.
            fade_ids = {id(e[0]) for e in getattr(self, '_label_fade_items', [])}
            self._remove_cohort_labels(keep_ids=fade_ids)

            # Smart labels: resolve duplicate labels across cohorts
            smart_label_mode = self.smart_label_mode_combo.currentText()
            smart_label_map = None
            if smart_label_mode != "Raw":
                try:
                    smart_label_map = self._apply_smart_labels(cluster_nodes)
                except Exception as e:
                    print(f"Smart labels failed, falling back to normal labels: {e}")
                    import traceback
                    traceback.print_exc()
                    smart_label_map = None

            self.cohort_label_items = []
            for cid, idx in cluster_nodes.items():
                # Compute centroid (center of cohort) from member indices
                centroid = scene.positions[idx].mean(axis=0)
                spread = float(self.spread_spin.value())
                centroid = centroid * spread

                # Compute dominant tags (filtered list)
                if smart_label_map is not None:
                    tags = smart_label_map.get(cid, [])
                else:
                    tags = self._compute_cohort_label_text(cid, nodes)
                # Fallback for selected cohort: ensure a label is always shown
                # even when no dominant tags qualify (e.g. all tags are shared
                # query tags or none meet the threshold).
                if not tags and cid == selected_cid:
                    tags = [f"Cohort {cid}"]
                if not tags:
                    continue

                # Determine label size
                if self.dynamic_label_size_checkbox.isChecked():
                    size = base_size + int((len(nodes) / max_count) * (base_size * 2))
                else:
                    size = base_size

                # Render the whole label as a SINGLE multi-line GLTextItem.
                # The base pyqtgraph GLTextItem draws its text via
                # QPainter.drawText(QPointF, str), which treats the whole
                # string as ONE line -- embedded newlines are NOT rendered as
                # line breaks (lines concatenate, e.g. "Tag1Tag2Tag3"). So we
                # use a GLTextItem subclass that overrides paint() to draw each
                # line separately, offset in SCREEN space from the single
                # projected world anchor. All lines share one world anchor and
                # are offset in screen space, so the stack stays locked to the
                # camera (no world-space drift as the camera rotates).
                try:
                    from PySide6.QtGui import QFont
                    font = QFont("Helvetica", size)
                    label_text = "\n".join(tags)
                    MultiLineTextItem = _get_multiline_text_item_class()
                    if cid == selected_cid:
                        # Selected cohort: color follows blink state (primary
                        # vs secondary blink color), always fully opaque.
                        label_rgba = self._get_selected_label_rgba()
                    else:
                        # Other cohorts: primary color, dimmed when a selection
                        # is active and "Dim Non-Selected" is on (mirrors the
                        # node dimming so the selection stands out).
                        label_rgba = (r, g, b, 255)
                        if selected_cid is not None and self.dim_non_selected_checkbox.isChecked():
                            label_rgba = (r, g, b, int(self.dim_alpha_spin.value() * 255))
                    label_item = MultiLineTextItem(
                        pos=centroid,
                        text=label_text,
                        color=label_rgba,
                        font=font,
                        alignment=Qt.AlignHCenter,
                    )
                    # Apply outline settings
                    from PySide6.QtGui import QColor as _QColor
                    label_item.outline_color = _QColor(*self._cohort_label_outline_color)
                    label_item.outline_width = float(self.cohort_label_outline_width_spin.value())
                    self.gl_view.addItem(label_item)
                    label_item.setVisible(True)
                    self.cohort_label_items.append(label_item)
                    self.cohort_label_map[cid] = label_item
                except Exception as e:
                    print(f"Error creating cohort label: {e}")

        except Exception as e:
            import traceback
            print(f"Error updating cohort labels: {e}")
            traceback.print_exc()

    def _remove_cohort_labels(self, keep_ids=None):
        """Remove cohort label items from the 3D view.

        Items that are mid-fade-out (tracked in ``_label_fade_items``) are always
        left in place so their fade timer can finish and remove them — this is what
        makes selection-switch fades actually visible instead of vanishing instantly.

        Args:
            keep_ids: optional extra set of item ids to leave in place.
        """
        if not hasattr(self, 'cohort_label_items'):
            self.cohort_label_items = []
            return
        fading_ids = {id(e[0]) for e in getattr(self, '_label_fade_items', [])}
        keep_ids = (keep_ids or set()) | fading_ids
        for item in self.cohort_label_items:
            if id(item) in keep_ids:
                continue
            try:
                self.gl_view.removeItem(item)
            except Exception:
                pass
        self.cohort_label_items = []
        self.cohort_label_map = {}
