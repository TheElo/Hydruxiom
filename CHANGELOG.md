# Changelog

## v0.7 — 2026-08-19, since V0.61

### Features
- **Pre-SVD before UMAP** (Algorithm group): new *Pre-SVD* toggle + components spinbox (default OFF, 64). Runs TruncatedSVD on the tag matrix to collapse e.g. 20k+ tag dims → ~64 dense dims before UMAP — since umap-learn densifies internally and neighbor-search cost scales with dimensionality, this makes the non-subsampled path dramatically faster (13.9s → 2.0s in local tests) and far lighter on RAM while preserving local structure well ([`Reducer.fit_transform`](src/pipeline/reducer.py))
- **Chunked Transform** (Algorithm group): new toggle (default ON), only active with Subsample — projects all points into the fitted UMAP space in bounded row chunks (~1.5 GB peak) instead of one giant call that would densify `files × tags` at once; results are identical, RAM is capped ([`Reducer.fit_transform`](src/pipeline/reducer.py))
- **Pre-flight RAM check** — before every Load & Compute / Recompute the pipeline estimates UMAP peak RAM (formulas in [`benchmarks/ram_simulator.py`](benchmarks/ram_simulator.py)) and prints a `[RAM check]` console line; if it will likely OOM you get a status-bar warning suggesting Subsample. Non-blocking: execution always continues
- **Label Space — overlap handling for cohort labels** (Cohort Label panel): new *Label Space* mode + gap control. Both modes are **live and smooth** — they re-solve continuously as you orbit, easing in/out rather than snapping ([`_apply_label_space`](src/ui/cohort_labels.py)):
  - **None**: no handling (legacy behavior)
  - **Fade** *(default)*: a non-selected label overlapping another fades to fully transparent while it does (so it can't block what's behind/in front of it); the selected cohort's label always keeps priority. Fades back in as soon as the overlap clears
  - **Move**: overlapping labels drift apart in screen space until they stop colliding — the selected cohort's label stays put, all others move around it and each other with weight (no teleporting). *Gap* (px) sets the minimum clearance
- **Auto-Deorphan per operation** (Settings → DBSCAN Optimizer): the single "Never / After Load / After Regroup" combo is replaced by three checkboxes — *After Load and Compute*, *After Regroup / Optimize*, *After Split*. Each op that can create orphans gets its own toggle; Pop never triggers it. Legacy settings values are migrated automatically ([`settings_persistence.py`](src/ui/settings_persistence.py))
- **TLS Verify toggle per client** (Settings → Clients): uncheck to skip certificate verification for MITM proxies / self-signed certs — disables `verify` on the shared requests session and silences urllib3's InsecureRequestWarning. Defaults to ON so normal users keep full TLS security ([`make_session`](src/data/clients.py))
- **API URL normalization** — you can now type your endpoint however it actually is: bare `192.168.1.2:16609`, with a middleware path prefix like `192.168.1.2:16609/hyapi` (reverse-proxy setups), or full URLs. Missing schemes default to http, fixing "Invalid URL '/get_services': No scheme supplied" ([`normalize_api_url`](src/data/clients.py))
- **Parallel tag loading** (Settings → Performance): file-tag fetching now runs concurrently — ~4 parallel API requests (~1.8× faster) and ~2 direct-DB connections (~1.7×), the sweet spots from local I/O benchmarks. New *API Load Threads* / *Direct-DB Load Threads* spinboxes (set to 1 for legacy sequential behavior); workers do pure I/O only, so shared state stays safe without locks ([`DataLoader.load_in_chunks`](src/data/loader.py))
- **Path-specific chunk sizes** (Settings → Clients): the single *Chunk Size* is split into *API Chunk Size* (default 8192 — network-bound, bigger requests win) and new *Direct-DB Chunk Size* (default 4096 — local SQLite, flat/fast from ~512 up); legacy settings migrate automatically ([`settings_dialog.py`](src/ui/settings_dialog.py))
- **Generate color scheme from a node's image** (Visuals → Color Scheme row): 🖋 pipette button fetches the selected single node's file from Hydrus in a background thread and extracts a palette via KMeans (*Colors* spinbox, default 19); 💾 saves it under your own name into settings.json, 🗑 deletes stored schemes. Hardcoded schemes can't be deleted; generated + custom schemes survive session save/load ([`_PaletteWorker`](src/ui/ui_construction.py), [`color_schemes.py`](src/pipeline/color_schemes.py))
- **Manual → Performance tab** — the in-app manual now explains what makes UMAP slow and which setting fixes it: Pre-SVD, Subsample, Chunked Transform, CPU Cores / Low RAM, Min Tag Frequency (incl. n/% units), API/Direct-DB load threads + chunk sizes, plus a note on GPU UMAP being Linux-only ([`manual_dialog.py`](src/ui/manual_dialog.py))

### Layout
- **WASD Navigation group moved** from the left sidebar to the right sidebar's **Visuals** tab (next to Camera Settings), freeing up space in the left panel
- **Explore button is now icon-style**: ▶ play / ■ stop glyphs instead of words, with a tooltip explaining click-to-stop

### Fixes
- **Color scheme not preserved after Recompute / Regroup / session load** — `SceneGraph.build_from_data` always assigns Pastel colors; every rebuild path that reads `scene.colors` (size/spread/transparency changes, in-place recluster refresh, session save) silently reverted to Pastel when another scheme was active. The active scheme is now synced into the scene before each base-scatter cache build ([`_sync_scene_colors_to_scheme`](src/ui/tag_map_3d_tab.py)), and the color scheme is stored in + restored from session snapshots
- **WASD "index 0 out of bounds" after Recompute** — two unguarded paths assumed a cohort exists: WASD key navigation picked targets via `np.where(...)[0][0]`, and the "Keep WASD labels visible" anchor used `np.bincount(non_noise).argmax()`, which raises exactly this error when every node is noise (the all-noise state after Recompute). Both are now guarded ([`tag_map_3d_tab.py`](src/ui/tag_map_3d_tab.py))
- **Session load could show an empty view** — the saved camera distance was restored without a sanity check; a session captured with the camera zoomed far out beyond the data (e.g. `distance ≈ 97,000` on a ~27-unit-wide scene) made every node sub-pixel and invisible after reload. Distances more than 50× the data diagonal now fall back to fit-to-data ([`tag_map_3d_tab.py`](src/ui/tag_map_3d_tab.py))
- **"Error redrawing WASD paths" spam at startup** — `_wasd_redraw_if_active()` read `self.selected_cluster_id` before it existed during early session load; the attribute is now accessed defensively
- **Auto-split ran pointlessly after Recompute** — Recompute leaves every node unclustered by design (use Regroup to cluster), but auto-split still started a cycle and logged "No cohorts to split". It's now skipped for that one completion ([`data_pipeline.py`](src/ui/data_pipeline.py))
- **Client details could be silently lost on OK** — `clients.json` was saved near the end of settings apply; if any later step threw (e.g. tag-score reload), the save never ran and the dialog had already closed, so entered client data appeared to not store. Clients are now persisted first thing in [`apply_settings`](src/ui/settings_dialog.py) with a visible error box on failure
- **Mouse wheel over spin boxes / combos now scrolls the panel** instead of silently changing their value — previously scrolling a dense settings area would "randomly change random settings" as the cursor crossed input fields ([`install_wheel_guard`](src/ui/tag_map_utils.py))
- **Window no longer gets stuck at the top of the screen on launch** — the geometry clamp now reserves room for the window frame (title bar), which Windows draws above the client area; previously a full-height restore pushed the caption off-screen
- **Auto-split could silently fail to start after Load & Compute** — it selected the target cohort via the heavy info-panel path, and if that threw or no worker actually started, `_auto_split_active` got stuck True with nothing happening. Now it selects directly, guards against a busy worker, verifies a worker is running, and aborts cleanly (with a visible status message) otherwise
- **Faded labels left ghost outlines** — in Label Space Fade mode the outline pass ignored fill alpha, so fully-faded labels still drew their stroke; the outline now follows the fill's alpha ([`gl_text_items.py`](src/ui/gl_text_items.py))

### Tweaks
- **"Min Doc Freq" renamed to "Min Tag Frequency"** (clearer wording) and made **unit-aware**: new *n / %* unit selector — `n` = absolute file count (range 0–100,000), `%` = percent of the loaded collection; each unit remembers its own value across switches, settings reloads and session loads ([`_resolve_min_doc_freq`](src/ui/ui_construction.py))
- **UI Scale moved to the top** of Settings → UI group (most-used control first)
- Subsample tooltip rewritten: explains that subsampling caps RAM but is *slower* than plain UMAP when the full fit would have fit in memory

---

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
