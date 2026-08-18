# Changelog

## v0.6.1 — 2026-08-19, since V0.60

### Fixes
- **Left/right sidebar overflow** — scrollable panels could grow wider than the viewport when a child had a large minimum-width hint, pushing widgets past the panel edge; page width is now clamped to the viewport ([`clamp_scroll_page_width`](src/ui/tag_map_utils.py))
- **Main window bottom cut off / "can't resize"** — restored window geometry was not clamped to the screen, so a size saved under one UI Scale could come back taller than the display and hide the progress/status bar; now capped + repositioned onto an available monitor ([`_clamp_widget_to_screens`](src/ui/settings_persistence.py))
- **Sidebar drag range widened** — left/right panels can now be dragged from 180px up to 700px (was 250–350 / 200–350); center view keeps a 320px minimum so it's never squeezed out
- **Off-screen monitor guard fixed** — the existing "re-center on unplugged monitor" check called `QCoreApplication.screens()` which doesn't exist in PySide6 (silently no-op'd); now uses `QGuiApplication`

### Docs & Settings Cleanup
- README: added a **"Required API permissions"** section — only *Search for and fetch files* is needed; *Manage pages* additionally required just for the Send-to-Tab button. Explains the 403 "you need at least one these permissions" error users hit with fresh keys
- Clients settings: **Files Dir / Thumbs Dir hidden** (unused — all images load via API); values stay in `clients.json` for a future thumbnails-from-disk feature. DB Dir kept, now labeled as Direct-DB-only; README corrected to match

---

## v0.6 (V0.60)
### Features
- **Manual window** — 📖 Help button opens an in-app reference for controls & shortcuts ([`manual_dialog.py`](src/ui/manual_dialog.py))
- **Node Sizing modes** — Distance (default), Screen-constant, Uniform single size, Auto-scale to view distance
- **Node Blending modes** — Normal Alpha (new default), Additive, Simple; applies live
- **View Background color picker** for the 3D view
- **New color schemes: Nature & Sci-Fi**; Pastel extended to 19 colors; dropdown shows each scheme's own colors as a preview
- **WASD Navigation settings** — line/letter color, label size, "Keep WASD labels visible" (anchors on largest cohort when nothing selected)
- **Explore "Size" mode** — tour visits biggest → smallest cohorts, loops
- **Tag Query grid layout** — configurable Columns × Rows in Settings
- **Editable UI Scale** — type any value 25–250%

### Layout & Settings Reorganization
- Settings tabs: General | UI | Clients | Shortcuts | Smart Scale (all scrollable)
- Chunk Size moved to Settings → Clients; Algorithm + Cluster groups moved to a new right-sidebar "Algorithm" tab; Orbit Speed + Wobble grouped as "Camera Settings"
- Left sidebar scrolls; action buttons pinned at bottom. Split/Pop/Cut row moved to left panel, renamed (Split group→Split, Pop cohort→Pop, Cut out→Cut)
- Icon-only buttons: ⚙ Settings, 📖 Help, ▶ Explore, 🚀 Optimize

### Fixes
- **Flashing white windows** in media viewer — image decoding moved to main thread; tiles hidden before detach (Qt was promoting them to top-level OS windows)
- Instant return from full-res zoom (grid is hidden, not rebuilt); mid-load cohort switches cancel stale thumbnail fetches
- Spread no longer double-applied after Split/Regroup/Optimize (stored raw, applied at render only)
- Explore tour re-targets when Spread changes; star twinkle syncs on every scene rebuild; highlight alpha follows node transparency

### Default Changes
Blending: Additive→Normal Alpha · Transparency 0.8→0.9 · Twinkle count 2000→4000 · Explore: path preview off, accel/decel 0.6→0.05, radius base 8.0→0.0, size factor 2.0→0.2, cycles 3→1, max orbit time 30s→15s, elevation 40°→35°

### Housekeeping
- New [`launch_hydruxiom.sh`](launch_hydruxiom.sh) (Linux launcher)
