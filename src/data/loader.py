"""Data loader for 3D Tag Space Visualization.

Connects to Hydrus client and loads file IDs and tags in chunks.
"""

from src.utils.utility_functions import ConnectToClient


class DataLoader:
    """Loads file and tag data from Hydrus client in chunks.

    Supports parallel loading (benchmarks/benchmark_api_io.py): the API path is
    network-bound (~4 concurrent requests optimal) while direct-DB is local and
    disk-bound (~2 connections). Workers do PURE I/O only — raw tags are returned
    to the calling thread, which applies the per-file transform sequentially as
    each chunk completes. This keeps shared state (e.g. TagInterner, which is not
    thread-safe) safe without locks.
    """

    def __init__(self, client, chunk_size=500, client_name=None, use_direct_db=False,
                 api_chunk_size=None, direct_chunk_size=None,
                 api_max_workers=1, direct_max_workers=1):
        """Initialize the data loader.

        Args:
            client: Hydrus API client instance
            chunk_size: Legacy single chunk size (default: 500). Used as fallback
                for both paths when the path-specific sizes are not given.
            client_name: Client identifier (HE, MN, MS, N) for direct DB path lookup
                AND for creating per-thread API clients in parallel mode.
            use_direct_db: Whether to use direct DB mode for tag loading (default: False)
            api_chunk_size: Files per request on the HTTP/API path (benchmark optimum ~8192).
            direct_chunk_size: Files per query on the direct-DB path (flat/fast >= 512; default 4096).
            api_max_workers: Concurrent API requests (benchmark sweet spot ~4).
            direct_max_workers: Concurrent SQLite connections (sweet spot ~2).
        """
        self.client = client
        self.chunk_size = chunk_size
        self.api_chunk_size = int(api_chunk_size or chunk_size)
        self.direct_chunk_size = int(direct_chunk_size or chunk_size)
        self.api_max_workers = max(1, int(api_max_workers))
        self.direct_max_workers = max(1, int(direct_max_workers))
        self.client_name = client_name
        self.use_direct_db = use_direct_db
        self.all_file_ids = []
        self.tag_data = {}  # file_id -> list of tags
        self._direct_db_dir = None  # Validated DB dir if direct mode available
        self._direct_mode_active = False

        # If direct DB mode requested, validate the client DB path
        if self.use_direct_db and self.client_name:
            from src.data.direct_db import get_client_db_path, is_valid_db_dir
            db_dir = get_client_db_path(self.client_name)
            if db_dir and is_valid_db_dir(db_dir):
                self._direct_db_dir = db_dir
                self._direct_mode_active = True
                print(f"[DataLoader] Direct DB mode active for {self.client_name} ({db_dir})")
            else:
                print(f"[DataLoader] Direct DB path not set/valid for {self.client_name}; falling back to API")

    def load_all_file_ids(self, search_tags=None, max_files=None, tag_service="auto2"):
        """Load all file IDs from the Hydrus client.

        Args:
            search_tags: Optional list of tags to filter files (default: None for all archived)
            max_files: Optional maximum number of files to load
            tag_service: Tag service name to use for searching (default: "auto2")

        Returns:
            list: List of all file IDs
        """
        if search_tags is None:
            search_tags = ["system:archive"]

        try:
            # Add limit to avoid loading too many files at once
            search_tags_with_limit = search_tags.copy()
            if max_files:
                search_tags_with_limit.append(f"system:limit is {max_files}")
            else:
                search_tags_with_limit.append("system:limit is 20000")

            # Pass tag service name (not key) to search_files
            # The API expects a single human-readable service name like "auto2", "local", "all known tags"
            self.all_file_ids = self.client.search_files(
                tag_service_name=tag_service,
                tags=search_tags_with_limit,
                file_sort_type=13
            )
            print(f"Loaded {len(self.all_file_ids)} file IDs using tag service: {tag_service}")
            return self.all_file_ids
        except Exception as e:
            print(f"Error loading file IDs: {e}")
            return []

    def load_tags_for_files(self, file_ids, tag_service="auto2", service_key=None, transform=None):
        """Load tags for a list of file IDs.

        Args:
            file_ids: List of file IDs to load tags for
            tag_service: Tag service name (default: "auto2")
            service_key: Pre-resolved service key (optional, avoids lookup)
            transform: Optional callable(file_id, tags) -> list | None applied
                to each file's tags BEFORE storage (e.g. filter + tokenize).
                Return None to skip the file entirely.

        Returns:
            dict: Dictionary mapping file_id to list of tags
        """
        if not file_ids:
            return {}

        # Direct DB mode: query the Hydrus client DB directly (much faster at scale)
        if self._direct_mode_active:
            try:
                from src.data.direct_db import direct_load_tags_for_files
                tags_dict = direct_load_tags_for_files(
                    file_ids, tag_service=tag_service, db_dir=self._direct_db_dir
                )
                self._apply_transform(tags_dict, transform)
                self.tag_data.update(tags_dict)
                return tags_dict
            except Exception as e:
                print(f"Error loading tags via direct DB: {e}; falling back to API")
                self._direct_mode_active = False  # Disable direct mode on failure

        try:
            metadata = self.client.get_file_metadata(file_ids=file_ids)

            # Get service key for the tag service (cache if provided)
            if service_key is None:
                from src.utils.query_comperator import get_service_key_by_name
                service_key = get_service_key_by_name(self.client, tag_service)

            if not service_key:
                print(f"Warning: Could not find service key for {tag_service}")
                service_key = '6c6f63616c2074616773'  # Default to local

            tags_dict = self._tags_from_metadata(metadata, service_key)

            self._apply_transform(tags_dict, transform)
            self.tag_data.update(tags_dict)
            return tags_dict
        except Exception as e:
            print(f"Error loading tags: {e}")
            return {}

    @staticmethod
    def _tags_from_metadata(metadata, service_key):
        """Parse get_file_metadata output into {file_id: [tag strings]}.

        Prefers display tags (siblings/parents applied), falling back through
        current display -> current storage -> pending display -> pending storage
        so freshly-imported files whose tags haven't materialised are still counted.
        Pure function — safe to call from worker threads.
        """
        tags_dict = {}
        for item in metadata:
            file_id = item['file_id']
            svc = item.get('tags', {}).get(service_key, {})
            storage = svc.get('storage_tags', {})
            display = svc.get('display_tags', {})
            tags = (display.get('0') or storage.get('0') or display.get('1') or storage.get('1') or [])
            tags_dict[file_id] = tags
        return tags_dict

    @staticmethod
    def _apply_transform(tags_dict, transform):
        """Apply a per-file transform to a chunk's tags dict in place.

        Args:
            tags_dict: dict mapping file_id -> list of tags (mutated in place)
            transform: callable(file_id, tags) -> list | None, or None to skip
        """
        if transform is None:
            return
        for fid in list(tags_dict.keys()):
            result = transform(fid, tags_dict[fid])
            if result is None:
                del tags_dict[fid]
            else:
                tags_dict[fid] = result

    def _resolve_service_key(self, tag_service):
        """Resolve + cache the service key once (avoids N get_services calls)."""
        from src.utils.query_comperator import get_service_key_by_name
        service_key = get_service_key_by_name(self.client, tag_service)
        if not service_key:
            print(f"Warning: Could not find service key for {tag_service}")
            service_key = '6c6f63616c2074616773'  # Default to local
        return service_key

    def _consume_chunk(self, chunk, raw_tags, transform, callback):
        """Apply the per-file transform on the CALLING thread and merge results.

        Must run outside worker threads: the transform may mutate shared state
        (TagInterner is not thread-safe) and emits Qt progress signals.
        """
        self._apply_transform(raw_tags, transform)
        self.tag_data.update(raw_tags)
        if callback:
            callback(chunk, raw_tags, len(self.tag_data))

    def load_in_chunks(self, callback=None, tag_service="auto2", search_tags=None, max_files=None, transform=None):
        """Load all data in chunks (sequential or parallel per path).

        Args:
            callback: Optional callback function(chunk_file_ids, chunk_tags, total_loaded)
                — invoked on the calling thread as each chunk completes.
            tag_service: Tag service name (default: "auto2")
            search_tags: Optional list of tags to filter files
            max_files: Optional maximum number of files to load
            transform: Optional per-file transform applied during load
                (see load_tags_for_files). Applied on the calling thread as each
                chunk completes, so it stays safe under parallel fetching.

        Returns:
            dict: Complete tag data for all files
        """
        # Load all file IDs first, passing tag_service for consistent search
        self.load_all_file_ids(search_tags, max_files, tag_service=tag_service)

        if not self.all_file_ids:
            print("No files to load")
            return {}

        total_files = len(self.all_file_ids)
        use_direct = self._direct_mode_active
        workers = self.direct_max_workers if use_direct else self.api_max_workers

        # Parallel API needs a client_name to build per-thread clients; without it
        # we cannot create independent sessions, so degrade to sequential.
        if not use_direct and workers > 1 and not self.client_name:
            print("[DataLoader] No client_name set; using sequential API loading")
            workers = 1

        chunk_size = self.direct_chunk_size if use_direct else self.api_chunk_size
        mode = "direct-DB" if use_direct else "API"
        print(f"Loading tags for {total_files} files in chunks of {chunk_size} via {mode}"
              + (f" ({workers} parallel)" if workers > 1 else " (sequential)"))

        # Direct-DB parallel probes with ONE real chunk first: opening the session
        # validates the DB files and loading a chunk validates service resolution.
        # On failure we fall back to the API path for the WHOLE load — same guarantee
        # as the sequential code, which disables direct mode on its first error.
        probe_tags = None
        if use_direct and workers > 1:
            from src.data.direct_db import DirectDBSession
            try:
                _probe_session = DirectDBSession(self._direct_db_dir, tag_service=tag_service)
                _first_chunk = self.all_file_ids[:chunk_size]
                probe_tags = _probe_session.load_tags(_first_chunk)
                _probe_session.close()
            except Exception as e:
                print(f"Direct-DB parallel probe failed ({e}); falling back to API path")
                use_direct = False
                workers = self.api_max_workers if self.client_name else 1
                chunk_size = self.api_chunk_size

        chunks = [self.all_file_ids[i:i + chunk_size] for i in range(0, total_files, chunk_size)]

        try:
            if use_direct and workers == 1:
                self._load_chunks_direct_sequential(chunks, tag_service, transform, callback)
            elif use_direct:
                self._load_chunks_direct_parallel(chunks[1:], probe_tags or {}, chunks[0],
                                                  tag_service, transform, callback, workers)
            elif workers == 1:
                self._load_chunks_api_sequential(chunks, tag_service, transform, callback)
            else:
                self._load_chunks_api_parallel(chunks, tag_service, transform, callback, workers)
        finally:
            pass

        print(f"Finished loading {len(self.tag_data)} files with tags")
        return self.tag_data

    # ------------------------------------------------------------------ sequential paths (legacy behavior)

    def _load_chunks_direct_sequential(self, chunks, tag_service, transform, callback):
        """Direct-DB, serial: ONE persistent session for the entire load.

        This avoids per-chunk reconnection and cache table reloads.
        """
        from src.data.direct_db import DirectDBSession
        try:
            direct_session = DirectDBSession(self._direct_db_dir, tag_service=tag_service)
        except Exception as e:
            print(f"Failed to create DirectDBSession: {e}; falling back to API")
            self._direct_mode_active = False
            self._load_chunks_api_sequential(chunks, tag_service, transform, callback)
            return
        try:
            for chunk in chunks:
                tags_dict = direct_session.load_tags(chunk)
                self._consume_chunk(chunk, tags_dict, transform, callback)
        finally:
            direct_session.close()

    def _load_chunks_api_sequential(self, chunks, tag_service, transform, callback):
        """API path, serial (the original behavior)."""
        # Cache service_key lookup ONCE before the chunk loop (optimization)
        service_key = self._resolve_service_key(tag_service)
        for chunk in chunks:
            try:
                metadata = self.client.get_file_metadata(file_ids=chunk)
                raw_tags = self._tags_from_metadata(metadata, service_key)
            except Exception as e:
                print(f"Error loading tags: {e}")
                raw_tags = {}
            self._consume_chunk(chunk, raw_tags, transform, callback)

    # ------------------------------------------------------------------ parallel paths (pure I/O in workers)

    def _load_chunks_api_parallel(self, chunks, tag_service, transform, callback, workers):
        """API path with N concurrent requests — one client/session per worker.

        Workers fetch + parse only; the transform runs on the calling thread as
        each chunk completes (see class docstring for why).
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from src.data.clients import connect_to_client

        service_key = self._resolve_service_key(tag_service)

        def _fetch(chunk):
            c = connect_to_client(self.client_name)  # per-thread client/session
            metadata = c.get_file_metadata(file_ids=chunk)
            return self._tags_from_metadata(metadata, service_key)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_fetch, ch): ch for ch in chunks}
            for fut in as_completed(futures):
                chunk = futures[fut]
                try:
                    raw_tags = fut.result() or {}
                except Exception as e:
                    print(f"Error loading tags (parallel API, {len(chunk)} ids): {e}")
                    raw_tags = {}
                self._consume_chunk(chunk, raw_tags, transform, callback)

    def _load_chunks_direct_parallel(self, chunks, probe_tags, first_chunk, tag_service,
                                     transform, callback, workers):
        """Direct-DB with N concurrent sessions (one SQLite connection per worker).

        ``probe_tags``/``first_chunk`` are the already-loaded first chunk from the
        pre-flight probe; it is consumed first so progress starts immediately.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from src.data.direct_db import DirectDBSession

        self._consume_chunk(first_chunk, probe_tags or {}, transform, callback)

        def _fetch(chunk):
            s = DirectDBSession(self._direct_db_dir, tag_service=tag_service)  # per-thread connection
            try:
                return s.load_tags(chunk)
            finally:
                s.close()

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_fetch, ch): ch for ch in chunks}
            for fut in as_completed(futures):
                chunk = futures[fut]
                try:
                    raw_tags = fut.result() or {}
                except Exception as e:
                    print(f"Error loading tags (parallel direct-DB, {len(chunk)} ids): {e}")
                    raw_tags = {}
                self._consume_chunk(chunk, raw_tags, transform, callback)

    def _load_tags_direct(self, file_ids, session, transform=None):
        """Load tags for a chunk using a persistent DirectDBSession (legacy helper)."""
        tags_dict = session.load_tags(file_ids)
        self._apply_transform(tags_dict, transform)
        self.tag_data.update(tags_dict)
        return tags_dict

    def get_tag_data(self):
        """Get the loaded tag data.

        Returns:
            dict: Complete tag data for all files
        """
        return self.tag_data

    def get_file_ids_with_tags(self):
        """Get file IDs that have at least one tag.

        Returns:
            list: List of file IDs with non-empty tag lists
        """
        return [fid for fid, tags in self.tag_data.items() if tags]
