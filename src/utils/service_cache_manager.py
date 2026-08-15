"""Service cache manager for storing and retrieving tag services metadata from all clients."""

import json
import os
import sqlite3
from typing import Dict, List, Optional

SERVICES_CACHE_FILE = "services_cache.json"

def get_cache_file_path() -> str:
    """Get the full path to the services cache file."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), SERVICES_CACHE_FILE)

def get_available_clients_from_db() -> List[str]:
    """
    Get list of available client names from the database.

    Returns:
        List[str]: List of client names found in the database.
    """
    try:
        from src.ui.settings_manager import load_settings
        settings = load_settings()
        db_path = settings.get("db_path", "mydb.db")

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get all client IDs from ClientSettings table
        cursor.execute("SELECT DISTINCT clientID FROM ClientSettings")
        results = cursor.fetchall()
        conn.close()

        return [row[0] for row in results if row[0]]

    except Exception as e:
        print(f"Error reading clients from database: {e}")
        # Fallback: no clients configured
        return []

def load_services_cache() -> Dict[str, Dict[str, str]]:
    """
    Load services cache from JSON file.

    Returns:
        Dict[str, Dict[str, str]]: Dictionary mapping client names to their tag services.
            Format: {client_name: {service_name: service_key}}
    """
    cache_path = get_cache_file_path()

    if not os.path.exists(cache_path):
        return {}

    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        return cache_data
    except Exception as e:
        print(f"Error loading services cache: {e}")
        return {}

def save_services_cache(cache_data: Dict[str, Dict[str, str]]) -> bool:
    """
    Save services cache to JSON file.

    Args:
        cache_data (Dict[str, Dict[str, str]]): Services cache data to save.

    Returns:
        bool: True if save was successful, False otherwise.
    """
    cache_path = get_cache_file_path()

    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving services cache: {e}")
        return False

def get_available_clients() -> List[str]:
    """Get list of known client names from database or fallback to defaults."""
    return get_available_clients_from_db()

def update_services_cache_for_client(client_name: str, client) -> bool:
    """
    Update services cache for a specific client.

    Args:
        client_name (str): Name of the client (e.g. "client1")
        client: Hydrus client instance

    Returns:
        bool: True if update was successful, False otherwise.
    """
    try:
        from src.utils.utility_functions import AvailableTagService

        services = AvailableTagService(client)

        if not services:
            print(f"No services found for client {client_name}")
            return False

        # Load existing cache
        cache_data = load_services_cache()

        # Update with new services
        cache_data[client_name] = services

        # Save updated cache
        return save_services_cache(cache_data)

    except Exception as e:
        return False

def update_all_services_cache() -> Dict[str, bool]:
    """
    Update services cache for all available clients.
    
    Returns:
        Dict[str, bool]: Dictionary mapping client names to update success status.
    """
    from src.utils.utility_functions import ConnectToClient

    results = {}
    clients_to_try = get_available_clients()
    failed = []
    
    for client_name in clients_to_try:
        try:
            client = ConnectToClient(client_name)
            success = update_services_cache_for_client(client_name, client)
            results[client_name] = success
            if not success:
                failed.append(client_name)
        except Exception:
            results[client_name] = False
            failed.append(client_name)
    
    # Print clean summary at the end
    successful = [c for c in clients_to_try if results.get(c)]
    
    if successful:
        print(f"[Services Cache] Connected to: {', '.join(successful)}")
    
    if failed:
        print(f"[Services Cache] Not available: {', '.join(failed)}")
    
    return results

def initialize_services_cache() -> bool:
    """
    Initialize services cache by attempting to connect to all clients on startup.
    
    Returns:
        bool: True if at least one client was successfully connected and cache updated,
              False if all connections failed.
    """
    print("[Startup] Checking client connections...")
    results = update_all_services_cache()
    success_count = sum(1 for success in results.values() if success)
    
    if success_count > 0:
        print(f"[Startup] Ready with {success_count} client(s) available")
        return True
    else:
        print("[Startup] Warning: No clients available. Using cached data if available.")
        return False

def get_tag_services_for_client(client_name: str) -> Optional[Dict[str, str]]:
    """
    Get tag services for a specific client from cache.

    Args:
        client_name (str): Name of the client (e.g. "client1")

    Returns:
        Optional[Dict[str, str]]: Dictionary mapping service names to service keys,
            or None if not found in cache.
    """
    cache_data = load_services_cache()
    return cache_data.get(client_name)

def get_all_cached_services() -> Dict[str, Dict[str, str]]:
    """
    Get all cached services from all clients.

    Returns:
        Dict[str, Dict[str, str]]: Dictionary mapping client names to their services.
    """
    return load_services_cache()