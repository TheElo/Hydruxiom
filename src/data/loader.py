"""Data loader for 3D Tag Space Visualization.

Connects to Hydrus client and loads file IDs and tags in chunks.
"""

from src.utils.utility_functions import ConnectToClient


class DataLoader:
    """Loads file and tag data from Hydrus client in chunks."""

    def __init__(self, client, chunk_size=500, client_name=None, use_direct_db=False):
        """Initialize the data loader.

        Args:
            client: Hydrus API client instance
            chunk_size: Number of files to load per chunk (default: 500)
            client_name: Client identifier (HE, MN, MS, N) for direct DB path lookup
            use_direct_db: Whether to use direct DB mode for tag loading (default: False)
        """
        self.client = client
        self.chunk_size = chunk_size
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
                search_tags_with_limit.append("system:limit is 100000")
            
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

            tags_dict = {}
            for item in metadata:
                file_id = item['file_id']
                # Prefer display tags (siblings/parents applied), falling back through
                # current display -> current storage -> pending display -> pending storage
                # so freshly-imported files whose tags haven't materialised are still counted.
                svc = item.get('tags', {}).get(service_key, {})
                storage = svc.get('storage_tags', {})
                display = svc.get('display_tags', {})
                tags = (display.get('0') or storage.get('0') or display.get('1') or storage.get('1') or [])
                tags_dict[file_id] = tags

            self._apply_transform(tags_dict, transform)
            self.tag_data.update(tags_dict)
            return tags_dict
        except Exception as e:
            print(f"Error loading tags: {e}")
            return {}

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

    def load_in_chunks(self, callback=None, tag_service="auto2", search_tags=None, max_files=None, transform=None):
        """Load all data in chunks.

        Args:
            callback: Optional callback function(chunk_file_ids, chunk_tags, total_loaded)
            tag_service: Tag service name (default: "auto2")
            search_tags: Optional list of tags to filter files
            max_files: Optional maximum number of files to load
            transform: Optional per-file transform applied during load
                (see load_tags_for_files)

        Returns:
            dict: Complete tag data for all files
        """
        # Load all file IDs first, passing tag_service for consistent search
        self.load_all_file_ids(search_tags, max_files, tag_service=tag_service)
        
        if not self.all_file_ids:
            print("No files to load")
            return {}

        total_files = len(self.all_file_ids)
        print(f"Loading tags for {total_files} files in chunks of {self.chunk_size}")

        # Cache service_key lookup ONCE before the chunk loop (optimization)
        from src.utils.query_comperator import get_service_key_by_name
        service_key = get_service_key_by_name(self.client, tag_service)
        if not service_key:
            service_key = '6c6f6c616c2074616773'  # Default to local

        # Process in chunks
        for start in range(0, total_files, self.chunk_size):
            end = min(start + self.chunk_size, total_files)
            chunk = self.all_file_ids[start:end]
            
            chunk_tags = self.load_tags_for_files(chunk, tag_service, service_key=service_key, transform=transform)
            
            if callback:
                callback(chunk, chunk_tags, len(self.tag_data))

        print(f"Finished loading {len(self.tag_data)} files with tags")
        return self.tag_data

    def get_tag_data(self):
        """Get the loaded tag data.

        Returns:
            dict: Dictionary mapping file_id to list of tags
        """
        return self.tag_data

    def get_file_ids_with_tags(self):
        """Get file IDs that have at least one tag.

        Returns:
            list: List of file IDs with non-empty tag lists
        """
        return [fid for fid, tags in self.tag_data.items() if tags]
