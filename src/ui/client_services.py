"""Client & tag-service management for the 3D tag map tab.

Mixin: methods operate on the ``TagMap3DTab`` instance (``self``). Moved here
from ``tag_map_3d_tab.py`` to reduce its size without changing behavior.
"""
import json
import os
import tempfile


class ClientServicesMixin:
    def _populate_tag_services(self, client_name):
        """Dynamically populate the tag service combo with all services
        available on the given Hydrus client.

        Uses client.get_services() which returns every registered tag service
        (name + key) directly — no file fetch required. Falls back to a
        sensible default list if the client is unreachable.
        """
        if not hasattr(self, 'tag_service_combo'):
            return

        # Preserve the current selection so we can restore it after repopulating
        previous = self.tag_service_combo.currentText()

        names = []
        if client_name:
            try:
                from src.utils.utility_functions import ConnectToClient
                client = ConnectToClient(client_name)
                services_dict = client.get_services() or {}
                # Only real TAG services belong in this combo. The flat
                # 'services' dict also contains file domains (my files, trash,
                # all local files...) and rating services,
                # which are not valid tag sources and would leak into the list.
                # Hydrus categorises them for us: use only the tag categories.
                tag_categories = ("local_tags", "all_known_tags", "tag_repositories")
                seen = set()
                collected = []
                for cat in tag_categories:
                    for info in services_dict.get(cat, []) or []:
                        if isinstance(info, dict):
                            nm = info.get("name", "")
                            if nm and nm not in seen:
                                seen.add(nm)
                                collected.append(nm)
                names = sorted(collected)
            except Exception as e:
                print(f"Could not fetch tag services for '{client_name}': {e}")

        # Fallback defaults if the client is unreachable or returned nothing
        if not names:
            names = ["auto2", "local", "all known tags"]

        self.tag_service_combo.blockSignals(True)
        self.tag_service_combo.clear()
        self.tag_service_combo.addItems(names)
        # Restore previous selection if it still exists, else first item
        idx = self.tag_service_combo.findText(previous)
        if idx >= 0:
            self.tag_service_combo.setCurrentIndex(idx)
        self.tag_service_combo.blockSignals(False)

    def open_settings_dialog(self):
        """Open the advanced settings window for the 3D tag map tab.

        Window size/position are persisted to the settings file under
        "settings_dialog_geometry" (survives save_settings() because it now
        starts from the existing file contents).
        """
        from src.ui.settings_dialog import TagMap3DSettingsDialog
        from src.ui.tag_map_utils import SETTINGS_FILE

        # F3 toggles the settings window: if one is already open, closing it counts
        # as "OK" (apply + save) rather than opening a second dialog. This fires
        # re-entrantly from within the modal exec() event loop.
        existing = getattr(self, '_settings_dialog', None)
        if existing is not None and existing.isVisible():
            try:
                existing.apply_settings()  # write values back to tab + save settings
            except Exception as e:
                print(f"Error applying settings on close: {e}")
            existing.accept()              # closes the dialog (exec returns)
            self._settings_dialog = None
            return

        dialog = TagMap3DSettingsDialog(self)
        self._settings_dialog = dialog
        # Restore last size/position (best effort; no-op if never saved)
        try:
            if os.path.exists(SETTINGS_FILE):
                with open(SETTINGS_FILE, 'r') as f:
                    self._restore_window_geometry(json.load(f), "settings_dialog_geometry", dialog)
        except Exception:
            pass
        dialog.exec()
        # Persist new size/position for next time
        try:
            settings = {}
            if os.path.exists(SETTINGS_FILE):
                with open(SETTINGS_FILE, 'r') as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    settings = loaded
            self._save_window_geometry(settings, "settings_dialog_geometry", dialog)
            settings_dir = os.path.dirname(SETTINGS_FILE) or '.'
            tmp_fd, tmp_path = tempfile.mkstemp(dir=settings_dir, suffix='.tmp')
            try:
                with os.fdopen(tmp_fd, 'w') as f:
                    json.dump(settings, f, indent=2)
                os.replace(tmp_path, SETTINGS_FILE)
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except Exception:
            pass

    def open_manual_dialog(self):
        """Open the manual (help) window. Non-modal; safe to reopen."""
        from src.ui.manual_dialog import ManualDialog
        dlg = ManualDialog(self)
        dlg.exec()

    def _refresh_client_combo(self):
        """Rebuild the client combo from clients.json, preserving selection.

        Called after the Settings window saves client changes so newly added /
        renamed / removed clients are reflected without restarting the app.
        """
        if not hasattr(self, 'client_combo'):
            return
        from src.data.clients import client_ids
        previous = self.client_combo.currentText()
        self.client_combo.blockSignals(True)
        self.client_combo.clear()
        ids = client_ids() or []
        self.client_combo.addItems(ids)
        idx = self.client_combo.findText(previous)
        if idx >= 0:
            self.client_combo.setCurrentIndex(idx)
        elif ids:
            self.client_combo.setCurrentIndex(0)
        self.client_combo.blockSignals(False)
        # Refresh tag services for the (possibly new) selected client.
        self._populate_tag_services(self.client_combo.currentText())
