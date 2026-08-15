# Hydruxiom

**3D Tag Space Explorer** — standalone app extracted from HydrusForHydrus.

Projects a Hydrus tag collection into navigable 3D space:
TF-IDF vectorization → UMAP/PCA reduction → DBSCAN clustering → 3D point cloud
with cohort exploration, relationship lines, and a synced media viewer.

## Status

- [x] Project skeleton created (2026-08-15)
- [ ] Migrate code from HydrusForHydrus (see `docs/migration.md`)
- [ ] Refactor monolithic `tag_map_3d_tab.py` into layered modules
- [ ] Standalone entry point (`main.py` → `src/app.py`)

## Structure

```
src/
├── core/       # Domain models, tag interner, config (pure Python, no Qt)
├── data/       # Hydrus integration: API client, direct DB, chunked loader
├── pipeline/   # ML: vectorizer, reducer (UMAP/PCA), clusterer (DBSCAN), scene builder
├── render/     # 3D rendering: scene view, scatter, hulls, relationships, labels
├── ui/         # Qt: main window, panels, media viewer, settings, workers
└── utils/      # Logging, paths
benchmarks/     # Headless performance benchmarks (UMAP cores, etc.)
tests/          # Unit tests for pipeline/core
docs/           # Architecture + migration notes
```

## Layering rule

`core` ← `data` ← `pipeline` ← `render` ← `ui`

Lower layers must never import from higher layers. `pipeline/` and `data/`
must be importable without Qt (enables headless benchmarks + unit tests).

## Run

```
launch_hydruxiom.bat
```
