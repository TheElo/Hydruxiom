from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt
from typing import Optional
from io import BytesIO
from PIL import Image, ImageQt

def get_image_ids_for_query(client, query_tags, limit, tag_service_name="all known tags"):
    """Centralized function to get image IDs for a query"""
    base_query = [f"system:limit is {limit}"]
    if isinstance(query_tags, str):
        query_tags = [query_tags]

    for tag in query_tags:
        base_query.insert(1, tag.strip())

    return client.search_files(
        tags=base_query,
        tag_service_name=tag_service_name,
        file_sort_type=13
    )

def load_pixmap_with_lanczos(image_bytes: bytes, max_size: int = 400) -> Optional[QPixmap]:
    """
    Load an image from bytes, resize with Pillow's LANCZOS filter while keeping aspect ratio
    following the same landscape/portrait/square rules from the original code,
    and return a QPixmap ready for Qt6 display.

    :param image_bytes: Raw image data as bytes.
    :param max_size: Target maximum width/height in pixels.
    :return: QPixmap or None if loading fails.
    """
    # Use QPixmap.loadFromData() which doesn't create temporary windows
    pixmap = QPixmap()
    if not pixmap.loadFromData(image_bytes):
        return None

    # Scale the pixmap to the target size using LANCZOS (smooth) transformation
    img_width = pixmap.width()
    img_height = pixmap.height()

    if img_width > img_height:
        new_height = int((max_size / img_width) * img_height)
        target_size = (max_size, min(new_height, max_size))
    elif img_width < img_height:
        new_width = int((max_size / img_height) * img_width)
        target_size = (min(new_width, max_size), max_size)
    else:
        target_size = (max_size, max_size)

    # Use Qt's smooth transformation for high-quality resizing
    scaled_pixmap = pixmap.scaled(
        target_size[0], target_size[1],
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation
    )

    return scaled_pixmap if not scaled_pixmap.isNull() else None

def load_images_for_window(window_instance, client, file_ids, limit=None, use_parallel=True):
    """
    Load images for either main or preview window with optimized bulk layout updates.

    Args:
        window_instance: Window instance with image_layout and other required attributes
        client: Hydrus client instance
        file_ids: List of file IDs to load
        limit: Maximum number of images to display (None = use all available slots)
        use_parallel: Whether to use parallel loading for better performance
    """
    if hasattr(window_instance, 'clear_images'):
        window_instance.clear_images()

    is_split_window = hasattr(window_instance, 'main_window') and window_instance.main_window

    if is_split_window:
        # For split window, use the instance's split window dimensions
        cols = getattr(window_instance, 'split_window_columns', 9)
        rows = getattr(window_instance, 'split_window_rows', 5)
    else:
        # For main window, use the instance's cols/rows attributes
        cols = getattr(window_instance, 'cols', 4)
        rows = getattr(window_instance, 'rows', 1)

    if limit is not None:
        total_slots = limit
    else:
        total_slots = cols * rows

    # Get resolution from settings (default to 400 if not found)
    if not hasattr(window_instance, '_cached_max_size'):
        from src.ui.settings_manager import load_settings
        settings = load_settings()

        if is_split_window:
            max_size = int(settings.get("split_window_resolution", 400))
        else:
            max_size = int(settings.get("main_window_resolution", 400))

        window_instance._cached_max_size = max_size
    else:
        max_size = window_instance._cached_max_size

    # Use bulk layout optimization for better performance
    from src.ui.layout.grid_layout_optimizer import GridLayoutOptimizer

    # Fill with black squares first using optimized bulk addition
    if hasattr(window_instance, 'create_black_square'):
        black_squares = [window_instance.create_black_square() for _ in range(total_slots)]
    else:
        from PySide6.QtWidgets import QLabel
        black_squares = []
        for _ in range(total_slots):
            label = QLabel()
            label.setFixedSize(max_size, max_size)
            label.setStyleSheet("background-color: black; border: 1px solid #333;")
            black_squares.append(label)

    GridLayoutOptimizer.add_widgets_in_bulk(window_instance.image_layout, black_squares, cols)

    if file_ids:
        # Disable updates on the entire window before loading images to prevent flickering
        window_instance.setUpdatesEnabled(False)
        
        try:
            if use_parallel and len(file_ids) > 1:
                # Use parallel loading for better performance with multiple images
                from src.utils.parallel_image_loader import load_images_for_window_parallel

                # Determine max workers based on number of images
                max_workers = None
                try:
                    import os
                    max_workers = min(os.cpu_count() or 4, len(file_ids))
                except:
                    max_workers = min(4, len(file_ids))

                load_images_for_window_parallel(
                    window_instance,
                    client,
                    file_ids[:total_slots],  # Only load as many as we can display
                    limit=limit,
                    max_workers=max_workers
                )
            else:
                # Use sequential loading for single image or when parallel is disabled
                for index, file_id in enumerate(file_ids[:total_slots]):
                    if hasattr(window_instance, 'load_image'):
                        grid_index = index % total_slots
                        row = grid_index // cols
                        col = grid_index % cols

                        window_instance.load_image(client, file_id, grid_index)

                        if index < len(black_squares):
                            black_square = black_squares[index]
                            black_square.setParent(None)
                            black_square.deleteLater()
        finally:
            # Re-enable updates after all images are loaded
            window_instance.setUpdatesEnabled(True)

def load_images_from_query_split_window(window_instance, query):
    """Load and display multiple images based on tag query specifically for split window"""
    # Convert single tag to query if needed
    if isinstance(query, str):
        query = [query]
    # Ensure query is a list
    if not isinstance(query, list):
        print("Split Window Error: Query must be a list of tags.")
        return

    cols_split = getattr(window_instance, 'split_window_columns', 9)  # Use split window specific settings
    rows_split = getattr(window_instance, 'split_window_rows', 5)  # Use split window specific settings
    # Ensure these values are properly set as instance attributes
    window_instance.split_window_columns = cols_split
    window_instance.split_window_rows = rows_split
    # Note: Do NOT set the regular cols and rows attributes to avoid conflicts with main window
    # The split window should only use its split_window_columns and split_window_rows attributes

    limit = cols_split * rows_split

    # Create base query with dynamic limit and image type
    base_query = [f"system:limit is {limit}"]
    # Add the current tag to the query
    for tag in query:
        base_query.insert(1, tag)

    # Check if query filtering is enabled (get from current tab)
    if hasattr(window_instance, 'main_window') and window_instance.main_window:
        current_tab = None
        if hasattr(window_instance.main_window, 'tab_manager'):
            current_tab = window_instance.main_window.tab_manager.get_current_tab()

        if current_tab and hasattr(current_tab, 'query_filter_images_enabled') and current_tab.query_filter_images_enabled:
            query_input = getattr(current_tab, 'query_input', None)
            current_query = ""
            if query_input and hasattr(query_input, 'text'):
                current_query = query_input.text().strip()
            if current_query:
                for tag in [t.strip() for t in current_query.split(',')]:
                    if tag and tag not in base_query:
                        base_query.insert(1, tag)

    print(f"Split Window Using Query: {base_query}")

    from src.utils.utility_functions import ConnectToClient

    # Get the current tab to use its client type setting
    current_tab = None
    if hasattr(window_instance, 'main_window') and window_instance.main_window:
        if hasattr(window_instance.main_window, 'tab_manager'):
            current_tab = window_instance.main_window.tab_manager.get_current_tab()

    # Use current tab's client type if available, otherwise fall back to main window
    if current_tab and hasattr(current_tab, 'client_type_combo') and hasattr(current_tab.client_type_combo, 'currentText'):
        client_type = current_tab.client_type_combo.currentText()
    else:
        client_type = getattr(window_instance.main_window, 'client_type', "")

    client = ConnectToClient(client_type)

    tag_service = current_tab.tag_service_file_tags if current_tab and hasattr(current_tab, 'tag_service_file_tags') else "all known tags"

    print(f"Loading images - Query: {base_query}, Tag Service: {tag_service}, Limit: {limit}")
    file_ids = client.search_files(tags=base_query,
                                  tag_service_name=tag_service,
                                  file_sort_type=13)

    print("Split Window Image Information:")
    print(f"Number of file IDs found: {len(file_ids)}")
    if len(file_ids) > 0:
        for index, file_id in enumerate(file_ids):
            row = index // cols_split
            col = index % cols_split
            print(f"{file_id} (R{row}, C{col})")
    else:
        print("No file IDs found - empty list returned")

    if hasattr(window_instance, 'clear_images'):
        window_instance.clear_images()

    # Use the updated load_images_for_window function with parallel loading enabled
    load_images_for_window(window_instance, client, file_ids, limit=limit, use_parallel=True)