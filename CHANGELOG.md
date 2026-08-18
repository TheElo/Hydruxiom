## Features
- **Manual window** — 📖 Help button opens an in-app reference for controls & shortcuts ([`manual_dialog.py`](src/ui/manual_dialog.py))
- **Node Sizing modes** — Distance (default), Screen-constant, Uniform single size, Auto-scale to view distance
- **Node Blending modes** — Normal Alpha (new default), Additive, Simple; applies live
- **View Background color picker** for the 3D view
- **New color schemes: Nature & Sci-Fi**; Pastel extended to 19 colors; dropdown shows each scheme's own colors as a preview
- **WASD Navigation settings** — line/letter color, label size, "Keep WASD labels visible" (anchors on largest cohort when nothing selected)
- **Explore "Size" mode** — tour visits biggest → smallest cohorts, loops
- **Tag Query grid layout** — configurable Columns × Rows in Settings
- **Editable UI Scale** — type any value 25–250%

## Layout & Settings Reorganization
- Settings tabs: General | UI | Clients | Shortcuts | Smart Scale (all scrollable)
- Chunk Size moved to Settings → Clients; Algorithm + Cluster groups moved to a new right-sidebar "Algorithm" tab; Orbit Speed + Wobble grouped as "Camera Settings"
- Left sidebar scrolls; action buttons pinned at bottom. Split/Pop/Cut row moved to left panel, renamed (Split group→Split, Pop cohort→Pop, Cut out→Cut)
- Icon-only buttons: ⚙ Settings, 📖 Help, ▶ Explore, 🚀 Optimize

## Fixes
- **Flashing white windows** in media viewer — image decoding moved to main thread; tiles hidden before detach (Qt was promoting them to top-level OS windows)
- Instant return from full-res zoom (grid is hidden, not rebuilt); mid-load cohort switches cancel stale thumbnail fetches
- Spread no longer double-applied after Split/Regroup/Optimize (stored raw, applied at render only)
- Explore tour re-targets when Spread changes; star twinkle syncs on every scene rebuild; highlight alpha follows node transparency

## Default Changes
Blending: Additive→Normal Alpha · Transparency 0.8→0.9 · Twinkle count 2000→4000 · Explore: path preview off, accel/decel 0.6→0.05, radius base 8.0→0.0, size factor 2.0→0.2, cycles 3→1, max orbit time 30s→15s, elevation 40°→35°

## Housekeeping
- New [`launch_hydruxiom.sh`](launch_hydruxiom.sh) (Linux launcher)
