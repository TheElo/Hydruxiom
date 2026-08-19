"""Hydrus client configuration.

Clients are defined in a JSON file (``clients.json``) at the project root,
NOT in an external SQLite DB. Each entry holds the API URL, API key, and the
local DB / files / thumbs directories used by direct-DB mode and the media
viewer.

Shape of clients.json::

    {
      "HE": {
        "label": "HE",
        "api_url": "http://127.0.0.1:<port>/",
        "api_key": "<64-char hex>",
        "db_dir": "<path to Hydrus client db folder>",
        "files_dir": "<path to files>",
        "thumbs_dir": "<path to thumbs>",
        "tls_verify": false
      },
      ...
    }

NOTE: clients.json contains API keys and local paths. It is gitignored.
"""

import json
import os
import shutil
from typing import Dict, List, Optional

# Project root = parent of src/
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLIENTS_FILE = os.path.join(_PROJECT_ROOT, "clients.json")


def clients_file_path() -> str:
    """Return the absolute path to clients.json."""
    return CLIENTS_FILE


def load_clients(path: Optional[str] = None) -> Dict[str, dict]:
    """Load all client configs from the JSON file.

    Args:
        path: Optional override path (defaults to the project clients.json).

    Returns:
        dict: client_id -> config dict. Empty dict if the file is missing.
    """
    p = path or CLIENTS_FILE
    if not os.path.exists(p):
        print(f"[Clients] No clients file at {p}")
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"[Clients] Error reading {p}: {e}")
        return {}


def save_clients(clients: Dict[str, dict], path: Optional[str] = None) -> bool:
    """Write client configs back to the JSON file (atomic).

    Keeps a ``.bak`` copy of the previous file before overwriting so a crash or
    bad edit can be recovered from.
    """
    p = path or CLIENTS_FILE
    try:
        if os.path.exists(p):
            shutil.copy2(p, p + ".bak")
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(clients, f, indent=2)
        os.replace(tmp, p)
        return True
    except Exception as e:
        print(f"[Clients] Error saving {p}: {e}")
        return False


def rename_client(old_id: str, new_id: str, path: Optional[str] = None) -> bool:
    """Rename a client ID (re-key the dict entry).

    Returns True on success. No-op (False) if old_id missing or new_id taken.
    """
    clients = load_clients(path)
    if old_id not in clients or new_id in clients:
        return False
    clients[new_id] = clients.pop(old_id)
    # Keep the label in sync with the new ID unless it was customized.
    cfg = clients.get(new_id, {})
    if not cfg.get("label") or cfg["label"] == old_id:
        cfg["label"] = new_id
    return save_clients(clients, path)


def client_ids() -> List[str]:
    """Return the list of configured client IDs (insertion order)."""
    return list(load_clients().keys())


def get_client_config(client_id: str) -> Optional[dict]:
    """Return the config dict for a client, or None if not configured."""
    return load_clients().get(client_id)


def get_client_db_dir(client_id: str) -> Optional[str]:
    """Return the Hydrus DB directory for a client (direct-DB mode).

    Returns None if the client is unknown or has no valid db_dir.
    """
    cfg = get_client_config(client_id)
    if not cfg:
        return None
    db_dir = (cfg.get("db_dir") or "").strip()
    if not db_dir:
        return None
    return db_dir


def normalize_api_url(api_url: str) -> str:
    """Normalize a user-entered Hydrus API base URL.

    Users may type the endpoint however their setup uses it, e.g.:
      - bare host+port:        ``192.168.1.40:16609``
      - with middleware path:  ``192.168.1.40:16609/hyapi``   (reverse proxy)
      - full URL:              ``http://127.0.0.1:45869/``

    This adds a scheme when missing (defaults to http, which covers local and
    LAN Hydrus instances / gateways) so that requests never fail with
    "Invalid URL '/get_services': No scheme supplied". Path prefixes are kept
    intact — hydrus-api appends endpoint paths by plain string concatenation.

    Returns the normalized base URL (no trailing slash; hydrus-api strips it).
    """
    import re
    from urllib.parse import urlsplit, urlunsplit

    u = (api_url or "").strip()
    if not u:
        return ""

    # Full URL with scheme — just clean up the path.
    m = re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", u)
    if m:
        parts = urlsplit(u)
        path = parts.path.rstrip("/")
        return urlunsplit((parts.scheme.lower(), parts.netloc or "localhost", path, "", ""))

    # No scheme — assume http. Split host[:port] from any trailing path manually
    # (urlsplit would misparse a bare "host:port" as path+query).
    head, _, tail = u.partition("/")
    netloc = head.strip() or "localhost"
    path = "/" + "/".join(p for p in tail.split("/") if p)  # drop empties/dupes
    return f"http://{netloc}{path}"


def make_session(tls_verify: bool = True):
    """Create a ``requests.Session`` for Hydrus API calls.

    When ``tls_verify`` is False, certificate verification is disabled and the
    urllib3 InsecureRequestWarning spam is suppressed. This exists because some
    users run behind MITM proxies or with self-signed certs; it defaults to ON
    so normal users keep full TLS security.
    """
    import requests

    session = requests.Session()
    if not tls_verify:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        session.verify = False
    return session


def connect_to_client(client_id: str):
    """Create a Hydrus API client for the given ID.

    Args:
        client_id: Client ID (e.g. "HE").

    Returns:
        hydrus_api.Client instance.

    Raises:
        KeyError: if the client is not configured.
    """
    cfg = get_client_config(client_id)
    if not cfg:
        raise KeyError(f"Client '{client_id}' not found in clients.json")

    import hydrus_api
    base = normalize_api_url(cfg["api_url"])
    session = make_session(cfg.get("tls_verify", True))
    # hydrus-api appends endpoint paths by string concatenation to api_url, so a
    # normalized base (scheme + optional path prefix like /hyapi) is all we need.
    return hydrus_api.Client(access_key=cfg["api_key"], api_url=base, session=session)
