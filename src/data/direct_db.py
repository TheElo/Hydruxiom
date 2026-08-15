"""Direct DB access module for Hydrus.

Provides drop-in replacements for Hydrus API functions that query the Hydrus
client DB directly instead of going through the API. Currently only the
metadata (tag loading) function is implemented, as benchmarks showed it is
~99% faster than the API at scale.

The direct functions take and return the same things the API functions do.
"""

import sqlite3
import os
from typing import List, Dict, Optional


def get_client_db_path(client_name: str) -> Optional[str]:
    """Get the Hydrus client DB directory path for a client.

    Hydruxiom: reads the ``db_dir`` field from clients.json instead of the
    old external mydb.db ClientSettings table.

    Args:
        client_name: Client identifier (e.g. "client1").

    Returns:
        The DB directory path if set and valid, else None.
    """
    from src.data.clients import get_client_config
    try:
        cfg = get_client_config(client_name)
        if not cfg:
            return None
        path = (cfg.get("db_dir") or "").strip()
        if not path or path.lower() == 'clientdbpath not set':
            return None
        return path
    except Exception as e:
        print(f"Error reading client DB path for {client_name}: {e}")
        return None


def set_client_db_path(client_name: str, db_dir: str) -> bool:
    """Set the Hydrus client DB directory path for a client.

    Hydruxiom: writes the ``db_dir`` field back to clients.json.

    Args:
        client_name: Client identifier (e.g. "client1").
        db_dir: Hydrus DB directory path.

    Returns:
        True if saved successfully.
    """
    from src.data.clients import load_clients, save_clients
    try:
        clients = load_clients()
        if client_name not in clients:
            clients[client_name] = {"label": client_name, "api_url": "", "api_key": "", "db_dir": "", "files_dir": "", "thumbs_dir": ""}
        clients[client_name]["db_dir"] = db_dir
        return save_clients(clients)
    except Exception as e:
        print(f"Error setting client DB path for {client_name}: {e}")
        return False


def is_valid_db_dir(db_dir: str) -> bool:
    """Check if a Hydrus DB directory is valid (contains client.db)."""
    if not db_dir or not os.path.isdir(db_dir):
        return False
    return os.path.exists(os.path.join(db_dir, "client.db"))


def _connect(db_dir: str):
    """Connect to all Hydrus DBs with ATTACH and Row factory."""
    conn = sqlite3.connect(os.path.join(db_dir, "client.db"))
    conn.row_factory = sqlite3.Row
    conn.execute(f"ATTACH DATABASE '{os.path.join(db_dir, 'client.mappings.db')}' AS mappings")
    conn.execute(f"ATTACH DATABASE '{os.path.join(db_dir, 'client.master.db')}' AS master")
    conn.execute(f"ATTACH DATABASE '{os.path.join(db_dir, 'client.caches.db')}' AS caches")
    return conn


def _get_service_id(conn, service_name: str) -> Optional[int]:
    """Get the service_id for a service name (e.g. 'local', 'all known tags')."""
    row = conn.execute(
        "SELECT service_id FROM services WHERE name = ?", (service_name,)
    ).fetchone()
    return row["service_id"] if row else None


def _resolve_service_id_with_mappings(conn, service_name: str) -> Optional[int]:
    """Resolve a service name to a service_id that has a current_mappings table.

    Virtual/aggregate services (e.g. 'all known tags', 'all known files') have
    no `current_mappings_{id}` table, so direct DB queries against them fail.
    Falls back to a real tag service ('local', then 'auto2') that does have a
    mappings table so direct DB mode can still be used.
    """
    service_id = _get_service_id(conn, service_name)
    if service_id is None:
        return None

    def _has_mappings(sid):
        table = f"current_mappings_{sid}"
        row = conn.execute(
            "SELECT 1 FROM mappings.sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        return bool(row)

    if _has_mappings(service_id):
        return service_id

    # Fall back to a real tag service that has a mappings table
    for fallback in ("local", "auto2"):
        fb_id = _get_service_id(conn, fallback)
        if fb_id is not None and _has_mappings(fb_id):
            return fb_id

    return None


def _resolve_display_tags(conn, raw_tag_ids_by_file, service_id):
    """Resolve raw storage tag_ids to display tag_ids (siblings + parents).

    For each file's raw tag_ids:
    1. Resolve siblings: replace bad_tag_id with ideal_tag_id.
    2. Expand parents: for each ideal tag, add all ancestor tags.

    Returns dict mapping hash_id -> list of resolved display tag_ids.
    """
    siblings_table = f"actual_tag_siblings_lookup_cache_{service_id}"
    parents_table = f"actual_tag_parents_lookup_cache_{service_id}"

    # Build sibling resolution map (bad_tag_id -> ideal_tag_id)
    sibling_rows = conn.execute(
        f"SELECT bad_tag_id, ideal_tag_id FROM caches.{siblings_table}"
    ).fetchall()
    sibling_map = {row["bad_tag_id"]: row["ideal_tag_id"] for row in sibling_rows}

    # Build parent expansion map (child_tag_id -> set of ancestor_tag_ids)
    parent_rows = conn.execute(
        f"SELECT child_tag_id, ancestor_tag_id FROM caches.{parents_table}"
    ).fetchall()
    parent_map = {}
    for row in parent_rows:
        parent_map.setdefault(row["child_tag_id"], set()).add(row["ancestor_tag_id"])

    display_tag_ids_by_file = {}
    for fid, raw_ids in raw_tag_ids_by_file.items():
        resolved = set()
        for tid in raw_ids:
            # Resolve sibling
            ideal = sibling_map.get(tid, tid)
            resolved.add(ideal)
            # Expand parents
            if ideal in parent_map:
                resolved.update(parent_map[ideal])
        display_tag_ids_by_file[fid] = list(resolved)

    return display_tag_ids_by_file


def direct_get_file_metadata_storage(file_ids, tag_service="local", db_dir=None):
    """Direct DB metadata returning only storage tags (no display resolution).

    Faster than direct_get_file_metadata (skips sibling/parent resolution).
    Returns list of dicts with keys: file_id, hash, tags (storage tags only).
    """
    if db_dir is None:
        raise ValueError("db_dir is required (no default path is configured)")
    conn = _connect(db_dir)
    try:
        service_id = _resolve_service_id_with_mappings(conn, tag_service)
        if service_id is None:
            return []
        mappings_table = f"current_mappings_{service_id}"

        placeholders = ','.join('?' for _ in file_ids)
        tag_rows = conn.execute(
            f"""
            SELECT m.hash_id, m.tag_id, n.namespace, s.subtag
            FROM mappings.{mappings_table} m
            JOIN master.tags t ON m.tag_id = t.tag_id
            LEFT JOIN master.namespaces n ON t.namespace_id = n.namespace_id
            JOIN master.subtags s ON t.subtag_id = s.subtag_id
            WHERE m.hash_id IN ({placeholders})
            """,
            file_ids,
        ).fetchall()

        tags_by_file = {}
        for row in tag_rows:
            namespace = row["namespace"]
            subtag = row["subtag"]
            tag_name = f"{namespace}:{subtag}" if namespace else subtag
            tags_by_file.setdefault(row["hash_id"], []).append(tag_name)

        hash_rows = conn.execute(
            f"""
            SELECT hash_id, hash FROM master.hashes
            WHERE hash_id IN ({placeholders})
            """,
            file_ids,
        ).fetchall()
        hash_by_id = {row["hash_id"]: row["hash"] for row in hash_rows}

        result = []
        for fid in file_ids:
            result.append({
                "file_id": fid,
                "hash": hash_by_id.get(fid, b""),
                "tags": tags_by_file.get(fid, []),
            })
        return result
    finally:
        conn.close()


def direct_get_file_metadata(file_ids, tag_service="local", db_dir=None):
    """Drop-in replacement for client.get_file_metadata.

    Args:
        file_ids: List of file IDs (hash_ids).
        tag_service: Tag service name (default 'local').
        db_dir: Hydrus DB directory (required).

    Returns:
        List of dicts with keys: file_id, hash, tags (matching API shape).
    """
    if db_dir is None:
        raise ValueError("db_dir is required (no default path is configured)")
    conn = _connect(db_dir)
    try:
        service_id = _resolve_service_id_with_mappings(conn, tag_service)
        if service_id is None:
            return []
        mappings_table = f"current_mappings_{service_id}"

        # Build tag_id -> tag_name map for the service
        placeholders = ','.join('?' for _ in file_ids)
        tag_rows = conn.execute(
            f"""
            SELECT m.hash_id, m.tag_id, n.namespace, s.subtag
            FROM mappings.{mappings_table} m
            JOIN master.tags t ON m.tag_id = t.tag_id
            LEFT JOIN master.namespaces n ON t.namespace_id = n.namespace_id
            JOIN master.subtags s ON t.subtag_id = s.subtag_id
            WHERE m.hash_id IN ({placeholders})
            """,
            file_ids,
        ).fetchall()

        # Collect raw tag_ids per file
        raw_tag_ids_by_file = {}
        for row in tag_rows:
            raw_tag_ids_by_file.setdefault(row["hash_id"], []).append(row["tag_id"])

        # Resolve to display tags (siblings + parents) for the service
        display_tag_ids_by_file = _resolve_display_tags(conn, raw_tag_ids_by_file, service_id)

        # Build tag_id -> tag_name map for resolved display tags
        all_resolved_ids = set()
        for ids in display_tag_ids_by_file.values():
            all_resolved_ids.update(ids)
        if all_resolved_ids:
            id_placeholders = ','.join('?' for _ in all_resolved_ids)
            name_rows = conn.execute(
                f"""
                SELECT t.tag_id, n.namespace, s.subtag
                FROM master.tags t
                LEFT JOIN master.namespaces n ON t.namespace_id = n.namespace_id
                JOIN master.subtags s ON t.subtag_id = s.subtag_id
                WHERE t.tag_id IN ({id_placeholders})
                """,
                list(all_resolved_ids),
            ).fetchall()
            tag_name_by_id = {}
            for row in name_rows:
                namespace = row["namespace"]
                subtag = row["subtag"]
                tag_name_by_id[row["tag_id"]] = f"{namespace}:{subtag}" if namespace else subtag
        else:
            tag_name_by_id = {}

        tags_by_file = {}
        for fid, resolved_ids in display_tag_ids_by_file.items():
            tags_by_file[fid] = [tag_name_by_id.get(tid, f"tag_id:{tid}") for tid in resolved_ids]

        # Get hashes for file IDs
        hash_rows = conn.execute(
            f"""
            SELECT hash_id, hash FROM master.hashes
            WHERE hash_id IN ({placeholders})
            """,
            file_ids,
        ).fetchall()
        hash_by_id = {row["hash_id"]: row["hash"] for row in hash_rows}

        result = []
        for fid in file_ids:
            result.append({
                "file_id": fid,
                "hash": hash_by_id.get(fid, b""),
                "tags": tags_by_file.get(fid, []),
            })
        return result
    finally:
        conn.close()


def direct_load_tags_for_files(file_ids, tag_service="local", db_dir=None):
    """Drop-in replacement for DataLoader.load_tags_for_files output.

    Returns a dict mapping file_id -> list of tags (same as the API path).
    """
    metadata = direct_get_file_metadata(file_ids, tag_service=tag_service, db_dir=db_dir)
    return {item["file_id"]: item["tags"] for item in metadata}
