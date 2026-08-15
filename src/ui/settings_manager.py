"""
Settings management module that handles loading, saving, and managing application settings.
"""

import json
from typing import Dict, Any

def load_settings(file_path: str = "tag_recommendation_settings.json") -> Dict[str, Any]:
    """
    Load settings from a JSON file with default values.

    Args:
        file_path (str): Path to the settings file. Defaults to "tag_recommendation_settings.json".

    Returns:
        Dict[str, Any]: Dictionary containing loaded settings.
    """

    with open(file_path, "r") as f:
        settings = json.load(f)
        return settings

def save_settings(settings: Dict[str, Any], file_path: str = "tag_recommendation_settings.json") -> None:
    """
    Save settings to a JSON file.

    Args:
        settings (Dict[str, Any]): Settings dictionary to save.
        file_path (str): Path to the settings file. Defaults to "tag_recommendation_settings.json".
    """
    try:
        with open(file_path, "w") as f:
            json.dump(settings, f)
    except IOError as e:
        print(f"Failed to save settings: {str(e)}")

def get_default_settings() -> Dict[str, Any]:
    """
    Get default settings dictionary.

    Returns:
        Dict[str, Any]: Dictionary containing default settings.
    """
    return {
        "last_query": "",
        "client_type": "",
        "tag_service_file_tags": "auto2",
        "tag_service_file_search": "local",
        "hide_scores": True,
        "show_hidden": True,
        "recommendation_limit": 500,
        "search_limit": 1200,
        "tag_threshold": 1,
        "grid_columns": 4,
        "grid_rows": 1,
        "window_width": 800,
        "window_height": 1200,
        "window_x": -1,
        "window_y": -1,
        "window_is_fullscreen": False,
        "query_filter_images": False,
        "db_path": "mydb.db",
        "screen_geometry": {},
        "blacklist": "set:*",
        "whitelist": "",
        "split_window_columns": 9,
        "split_window_rows": 5,
        "split_window_resolution": 400,
        "split_window_width": None,
        "split_window_height": None,
        "split_window_x": None,
        "split_window_y": None,
        "main_window_resolution": 400,
        "image_preview_visible": True,
        "last_selected_tag": "",
        "open_tabs": []
    }

def save_tabs_settings(tabs_data, file_path="tabs_settings.json"):
    """
    Save open tabs configuration to separate JSON file.

    Args:
        tabs_data (dict): Dictionary containing tabs configuration
        file_path (str): Path to the settings file. Defaults to "tabs_settings.json".
    """
    try:
        with open(file_path, "w") as f:
            json.dump(tabs_data, f, indent=2)
    except IOError as e:
        print(f"Failed to save tabs settings: {str(e)}")

def load_tabs_settings(file_path="tabs_settings.json"):
    """
    Load open tabs configuration from JSON file.

    Args:
        file_path (str): Path to the settings file. Defaults to "tabs_settings.json".

    Returns:
        dict: Dictionary containing loaded settings, or empty dict if file not found.
    """
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"open_tabs": [], "last_active_tab": None}
    except json.JSONDecodeError:
        print("Error parsing tabs_settings.json, using defaults")
        return {"open_tabs": [], "last_active_tab": None}