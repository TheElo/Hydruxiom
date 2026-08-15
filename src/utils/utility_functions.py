"""
here I will collect functions for hydrus that proven generally useful for all kinds of stuff
"""

import sqlite3, hydrus_api
from src.ui.settings_manager import load_settings

# our client and settings management

def get_db_path():
    """Get the database path from settings with fallback to default"""
    settings = load_settings()
    return settings.get("db_path", "mydb.db")

def ConnectToClient(arc):
    """Create a Hydrus API client from the clients.json config.

    Hydruxiom: client credentials (API URL + key) come from clients.json
    instead of the old external mydb.db ClientSettings table.
    """
    global client
    from src.data.clients import get_client_config
    cfg = get_client_config(arc)
    if not cfg:
        raise KeyError(f"Client '{arc}' not found in clients.json")
    client = hydrus_api.Client(access_key=cfg["api_key"], api_url=cfg["api_url"])
    return client

def AvailableClients():
    available_clients = []

    from src.data.clients import client_ids
    known_clients = client_ids() or []

    for arc in known_clients:
        try:
            client = ConnectToClient(arc)
            print(arc, client.get_api_version())
            available_clients.append(arc)
        except:
            continue

    return available_clients

def AvailableTagService(client):
    """
    Returns a dictionary of available tag services from Hydrus client metadata.

    Args:
        client: Hydrus API client instance

    Returns:
        dict: Dictionary mapping service names to their keys
    """
    try:
        file_ids = client.search_files(tags=["system:limit is 1"], file_sort_type=13)

        if not file_ids:
            print("No files found")
            return {}

        metadata = client.get_file_metadata(file_ids=[file_ids[0]])

        if not metadata or 'tags' not in metadata[0]:
            print("No tag information in metadata")
            return {}

        tag_services = {}
        for service_key, service_data in metadata[0]['tags'].items():
            try:
                service_name = client.get_services()['services'][service_key]['name']
                tag_services[service_name] = service_key
            except Exception as e:
                print(f"Error getting service name for key {service_key}: {e}")
                continue

        return tag_services

    except Exception as e:
        return {}

def AvailableTagServiceFromCache(client_name):
    """
    Returns cached tag services for a specific client.

    Args:
        client_name (str): Name of the client (e.g. "client1")

    Returns:
        dict: Dictionary mapping service names to their keys, or empty dict if not cached
    """
    from src.utils.service_cache_manager import get_tag_services_for_client
    return get_tag_services_for_client(client_name) or {}

def get_service_key_by_name(client, service_name):
    """
    Get service key by service name, checking cache first for better performance.

    Args:
        client: Hydrus client instance
        service_name (str): Name of the service to look up

    Returns:
        str: Service key if found, None otherwise
    """
    # First try to get from cache (faster, no API call)
    try:
        from src.utils.service_cache_manager import get_all_cached_services
        cached_services = get_all_cached_services()

        # Search through all cached clients
        for client_name, services in cached_services.items():
            if service_name in services:
                return services[service_name]
    except Exception as e:
        print(f"Error getting service key from cache: {e}")

    # Fallback to client API call if not in cache
    try:
        services_dict = client.get_services()
        service_key = None
        for key, service_info in services_dict['services'].items():
            if service_info['name'] == service_name:
                service_key = key
                break
        return service_key
    except Exception as e:
        print(f"Error getting service key from client: {e}")
        return None
