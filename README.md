<p align="center">
  <img src="icon/hydruxiom_256.png" width="256" alt="Hydruxiom icon">
</p>

<h1 align="center">Hydruxiom — 3D Tag Space Explorer</h1>

Standalone desktop app that projects a [Hydrus](https://hydrusnetwork.github.io/hydrus/) tag collection into navigable 3D space.

**Pipeline:** TF-IDF vectorization → UMAP/PCA reduction → DBSCAN clustering → interactive 3D point cloud with cohort exploration, relationship lines, and a synced media viewer.

Built with PySide6 + pyqtgraph (OpenGL). Extracted from the HydrusForHydrus plugin as an independent project.

---

## Quick Start (Windows)

1. Install **Python 3.10+** (check "Add to PATH" during install)
2. Clone this repo
3. Double-click **`launch_hydruxiom.bat`**

The launcher handles everything: it creates a `.venv/` on first run, installs/updates dependencies from `requirements.txt` on every launch, and starts the app. The window only stays open on crash so you can read the traceback.

### Manual setup (alternative)

```bat
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python main.py
```

---

## First Run: Configure a Client

The app needs at least one Hydrus client with the **API enabled** (Hydrus → Options → Network → API).

1. Start the app, press **F3** to open Settings
2. In the **Clients** section: **+ Add**, then fill in:
   - **Label / ID** — your name for the client
   - **API URL** — e.g. `http://127.0.0.1:4566/`
   - **API Key** — from Hydrus's API settings page (no masking; stored locally)
    - **DB Dir** *(optional)* — Hydrus client DB folder; only used by *Direct DB mode*. Leave empty if you use the API (the default)
3. Use **Test Connection** to verify, then **OK**

Client config is stored in `clients.json` at the project root (gitignored; a `.bak` copy is kept before every save).

### Required API permissions


| Permission | Needed for |
|---|---|
| ✅ **Search for and fetch files** | Everything core: loading the tag map (`search_files`, `get_file_metadata`), media viewer thumbnails/full-res images, Test Connection |
| ⬜ Manage pages *(optional)* | Only the **"Send to Tab"** button (Ctrl+T) — it lists Hydrus tabs and adds selected files to one of them |

Tick **Search for and fetch files** at minimum; add **Manage pages** only if you want Send-to-Tab. No other permissions are used — Hydruxiom never edits tags, ratings, notes or relationships in your collection.

If loading fails with a 403 / *"You need at least one these permissions…"* error, edit the key (Hydrus → Options → Network → API) and tick the missing box above.

---

## Usage

1. Pick a client (left panel → Client)
2. Enter a tag query — supports Hydrus syntax: `tag1, -excluded, [or-tag-a, or-tag-b]`
   The clickable tag grid in **Filter Settings** builds the query for you (click cycles: neutral → included → excluded → OR)
3. Press **F5 / Load & Compute** — loads file IDs + tags, runs TF-IDF → UMAP → DBSCAN
4. Explore: click nodes to select files, right-click clusters, use the media viewer (**F4**) for synced thumbnails

### Explore Mode (Helicopter Orbit)

Press the **Explore** button to fly the camera on an automated tour of your cohorts. The camera accelerates toward each target, banks smoothly into a circular orbit (with momentum — no stops or hard turns), circles it for a configurable number of laps, then flies on to the next cohort.

- **Mode:** Random (shuffled), Linear Path (spatial sweep), or Contrast (farthest-point sampling)
- **Orbit Speed / Cycles / Elevation / Radius** — all tunable in Settings → Explore
- A bright red marker shows which cohort is being orbited; an optional path preview draws the planned route
- While running, press **E** to jump to the next cohort or **Q** to go back to the previous one (the camera banks into each new orbit with momentum). Press **W** / **S** to zoom closer / farther from the current cohort.

### WASD Cohort Navigation

With a scene loaded, press **W/S/A/D** to jump between neighboring cohorts (screen-relative directions). Preview arrows show where each key will take you. **Q/E** step back/forward through your travel history, leaving a persistent trail.

### Keyboard shortcuts

| Key | Action |
|-----|--------|
| F3 | Settings window |
| F4 | Toggle media viewer |
| F5 | Load & Compute (full pipeline) |
| F6 | Recompute (UMAP only, keeps clusters) |
| F7 | Regroup (DBSCAN only, keeps positions) |
| F12 | 4x supersampled screenshot → `screenshots/` |
| Ctrl+X | Clear session (frees memory) |
| Ctrl+S | Split group (re-cluster selection into sub-cohorts) |
| Ctrl+E | Cut out selected cohort |
| Ctrl+T | Send selected files to a Hydrus tab |

Ctrl+ combos are skipped while a text field has focus.

| W/S/A/D | Navigate to nearest cohort in that screen direction |
| Q / E | Step back / forward through travel history (or previous / next Explore cohort while Explore is running) |

### Sessions & persistence

- Every successful load/recompute/pop **auto-saves** the scene to `sessions/latest.npz` (positions, tags, clusters, settings, camera)
- Enable **"Auto load session"** to skip straight to the last scene on startup (no reprocessing)
- Settings (incl. window positions/sizes and media-viewer open state) persist in `3d_tag_map_settings.json`, written atomically with a 2 s debounce so they survive crashes

### Performance notes

- **Direct DB mode** reads tags straight from Hydrus's SQLite files instead of the API — much faster for large collections (configure per-client dirs in Settings → Clients)
- **UMAP subsampling** (on by default, 70K subset): fits UMAP on a random subset, then transforms all points — makes multi-million-file collections feasible
- **Smart Scale:** automatically picks optimal UMAP/DBSCAN/visualization parameters based on file count (Settings → Smart Scale tab)

---

## Requirements

- Windows (developed/tested on Win 10/11)
- Python 3.10+
- A running Hydrus client with the API enabled
- See `requirements.txt` for pinned dependencies (PySide6, pyqtgraph, umap-learn, scikit-learn, hydrus-api, …)
