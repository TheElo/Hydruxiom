# This script connects to a configurable Hydrus client, executes multiple queries to find files,
# compares tag distributions between the queries, and shows which tags are more or less common.

from src.utils.utility_functions import ConnectToClient
from collections import defaultdict
import sqlite3
import fnmatch
import re

# Module-level variable to store external tag scores.
# OPTIONAL in Hydruxiom: populated only when a score DB path is configured in
# the 3D settings window (key "score_db_path"). Empty by default -> the
# vectorizer / scene graph apply no score weighting.
ExternalTagScores = {}

# Cache for API service responses - maps client instances to their services data
_services_cache = {}


def _score_db_path_from_settings():
    """Read the optional score DB path from the 3D tag map settings file."""
    import os
    try:
        settings_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "3d_tag_map_settings.json",
        )
        if os.path.exists(settings_file):
            import json
            with open(settings_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return (data.get("score_db_path") or "").strip()
    except Exception:
        pass
    return ""


def reload_external_tag_scores(db_path=None):
    """(Re)load ExternalTagScores from a SQLite DB with an ExternalTagScores table.

    Args:
        db_path: Path to the SQLite file. If None, reads "score_db_path" from
            the 3D settings file. Empty/None path -> clears scores (disabled).

    Returns:
        int: number of scores loaded.
    """
    global ExternalTagScores
    ExternalTagScores = {}
    if db_path is None:
        db_path = _score_db_path_from_settings()
    if not db_path:
        return 0
    try:
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT tag, score FROM ExternalTagScores")
        for tag, score in cursor.fetchall():
            ExternalTagScores[tag] = score
        conn.close()
        print(f"[TagScores] Loaded {len(ExternalTagScores)} scores from {db_path}")
    except Exception as e:
        print(f"[TagScores] Error loading scores from {db_path}: {e} (scoring disabled)")
        ExternalTagScores = {}
    return len(ExternalTagScores)


# Best-effort load at import time (never raises; no-op if no path configured).
try:
    reload_external_tag_scores()
except Exception:
    pass

def get_tags_for_files(client, file_ids, tag_service="all known tags", blacklist=None, whitelist=None):
    """
    Get tags for a list of file IDs.

    Args:
        client: Hydrus API client instance
        file_ids (list): List of file IDs
        tag_service (str): Tag service to use (e.g., "all known tags", "my tags")
        blacklist (list): List of patterns to exclude from results

    Returns:
        dict: Dictionary mapping file IDs to their tags
    """
    try:
        tag_dict = {}
        # Get metadata directly without chunking
        metadata = client.get_file_metadata(file_ids=file_ids)
        for item in metadata:
            file_id = item['file_id']
            service_key = get_service_key_by_name(client, tag_service)
            if not service_key:
                print(f"Warning: Could not find service key for {tag_service}, using default")
                service_key = '6c6f63616c2074616773'  # Default to local

            tags = item['tags'].get(service_key, {}).get('storage_tags', {}).get('0', [])

            # Apply whitelist or blacklist to tags
            if whitelist:
                # Pre-compile whitelist patterns for better performance
                whitelist_patterns = [re.compile(fnmatch.translate(pattern)) for pattern in whitelist]
                filtered_tags = [tag for tag in tags if any(p.match(tag) for p in whitelist_patterns)]
            elif blacklist:
                # Pre-compile blacklist patterns for better performance
                blacklist_patterns = [re.compile(fnmatch.translate(pattern)) for pattern in blacklist]
                filtered_tags = [tag for tag in tags if not any(p.match(tag) for p in blacklist_patterns)]
            else:
                # If neither is provided, use all tags as they are
                filtered_tags = tags

            tag_dict[file_id] = filtered_tags

        return tag_dict
    except Exception as e:
        print(f"Error getting tags: {e}")
        return {}

def get_service_key_by_name(client, service_name):
    """
    Get the service key for a given service name.

    Args:
        client: Hydrus API client instance
        service_name (str): Name of the service

    Returns:
        str: Service key or None if not found
    """
    try:
        # Use cached services data if available for this specific client
        global _services_cache
        if client not in _services_cache:
            _services_cache[client] = client.get_services()
        services_dict = _services_cache[client]
        for key, service_info in services_dict['services'].items():
            if service_info['name'] == service_name:
                return key
        print(f"Warning: Service '{service_name}' not found")
        return None
    except Exception as e:
        print(f"Error getting service key: {e}")
        return None

def count_tags(tag_dict):
    """
    Count occurrences of each tag across all files.

    Args:
        tag_dict (dict): Dictionary mapping file IDs to tags

    Returns:
        dict: Dictionary mapping tags to their counts
    """
    tag_counts = defaultdict(int)
    for tags in tag_dict.values():
        for tag in tags:
            tag_counts[tag] += 1
    return tag_counts

def analyze_query(client, query , tag_service_file_tags="my tags", blacklist=None, whitelist=None, tag_service_file_search="my tags"):
    """
    Analyze a single query and return tag statistics.

    Args:
        client: Hydrus API client instance
        query (list): Query to execute
        tag_service_file_tags (str): Tag service for file tags
        blacklist (list): List of patterns to exclude from results

    Returns:
        tuple: (file_count, tag_counts)
    """
    print(f"\nAnalyzing query: {query}")
    file_ids = client.search_files(tag_service_name=[tag_service_file_search], tags=query, file_sort_type=13)

    if not file_ids:
        print("No files found matching the query")
        return 0, {}

    total_files = len(file_ids)
    print(f"Found {total_files} files")

    tag_dict = get_tags_for_files(client, file_ids, tag_service=tag_service_file_tags,
                                 blacklist=blacklist, whitelist=whitelist)
    tag_counts = count_tags(tag_dict)

    del tag_dict

    return total_files, tag_counts

def compare_queries(query1, query2, tag_service_file_tags="all known tags", blacklist=None, whitelist=None, limit=100, tag_service_file_search="all known tags", client_type=""):
    """
    Compare tag distributions between two queries.

    Args:
        query1 (list): First query to analyze
        query2 (list): Second query to analyze
        tag_service_file_tags (str): Tag service for file tags
        blacklist (list): List of patterns to exclude from results

    Returns:
        list: List of tuples containing comparison data for each tag
    """
    # Connect to the specified client
    print(f"Connecting to {client_type} client...")
    client = ConnectToClient(client_type)

    if not client:
        print(f"Failed to connect to {client_type} client")
        return []
    
    # Analyze both queries
    total1, counts1 = analyze_query(client, query1, tag_service_file_tags, blacklist, whitelist,tag_service_file_search)
    total2, counts2 = analyze_query(client, query2, tag_service_file_tags, blacklist, whitelist,tag_service_file_search)

    # Get all unique tags from both queries
    all_tags = set(counts1.keys()).union(set(counts2.keys()))

    # Prepare comparison data
    comparison = []
    for tag in sorted(all_tags):
        count1 = counts1.get(tag, 0)
        count2 = counts2.get(tag, 0)
        commonality1 = count1 / total1 if total1 > 0 else 0
        commonality2 = count2 / total2 if total2 > 0 else 0
        delta = commonality2 - commonality1  # Store signed delta to show direction

        comparison.append((tag, count1, count2, commonality1, commonality2, delta))

    # Sort by delta (absolute difference in commonality)
    comparison.sort(key=lambda x: x[5], reverse=True)

    return comparison[:limit]  # Return only top 100 by delta

def RecommendTags(query=[], limit=50, search_limit=300, tag_service_file_search="my tags", tag_service_file_tags = "my tags", whitelist=None, blacklist=None, tag_threshold=3, client_type=""):
    """
    This function has the goal to return tag recommandations in a list sorted by most probable at the start.

    Notes:
    - dont recommend existing tags (tags from the query), add them to the blacklist when quering
    :param query: ["tagA", "TagB"] expects a single tags or a dict of tags to perform as a reference query. Later we want to cut the query if no recommended tags are found")
    :param limit: how many top tags to return, integer
    :param tag_service_file_search:
    :param tag_service_file_tags:
    :param whitelist:
    :param blacklist:
    :param tag_threshold: Number of tags in the query above which we don't extend query1 with anti_query
    :return:
    """

    anti_query = []
    for tag in query:
        # if string starts with "-" then remove the "-" prefix instead
        if tag.startswith("-"):
            string = tag[1:]  # Remove the "-" prefix
        else:
            string = "-" + tag
        anti_query.append(string)
    print("anti query: ", anti_query)

    # Parse query to handle OR syntax: [tag1, tag2, tag3]
    parsed_query = parse_query_tags_from_string(query)
    print("parsed query: ", parsed_query)

    # Create query1 based on tag count
    query1 = ["system:archive", f"system:limit is {search_limit}"]
    # query1 = [f"system:limit is {search_limit}"] # this was generally worse than with archive filter, we maybe should even add a min tag to the query
    if len(query) <= tag_threshold:
        # For small queries, extend with anti_query for more specific results
        query1.extend(anti_query)
    print("query1:", query1)

    query2 = [f"system:limit is {search_limit}"]
    # Check if query is an OR query (wrapped in brackets)
    if isinstance(query, str) and (query.startswith('[') and query.endswith(']')):
        # Parse OR query: [tag1, tag2, tag3] -> [["tag1", "tag2", "tag3"]]
        import re
        or_match = re.match(r'^\s*\[(.*)\]\s*$', query)
        if or_match:
            content = or_match.group(1).strip()
            if content.startswith('"') and content.endswith('"'):
                try:
                    import json
                    tags = json.loads(f'[{content}]')
                    query2.append(tags)
                except:
                    tags = [tag.strip().strip('"') for tag in content.split(',')]
                    query2.append(tags)
            else:
                tags = [tag.strip().strip('"') for tag in content.split(',')]
                query2.append(tags)
    else:
        query2.extend(query)
    print("query2:", query2)

    if not blacklist and not whitelist: # default blacklist fallback
        blacklist = ["set:*", "comic:*", "marker:*", "page:*", "page#:*"]

    if blacklist is not None:
        blacklist.extend(query) # adding query to blacklist so they don'T get recommended as existing tags are already set
    comparison = compare_queries(query1, query2, tag_service_file_tags, blacklist, whitelist, limit=limit, tag_service_file_search=tag_service_file_search, client_type=client_type)
    # comparison = compare_queries(query1, query2, tag_service_file_tags, blacklist, whitelist, limit=limit)

    if not comparison:
        print("Failed to compare queries")
        return
    """
    # Display the comparison table (commented out for performance)
    print("\nTop Tags by Commonality Delta:")
    print(f"{'Tag':<40} {'Count1':<8} {'Count2':<8} {'Commonality1':<15} {'Commonality2':<15} {'Delta':<15}")
    print("-" * 95)
    
    for row in comparison[:3]:  # Only show first 3 tags
        tag, count1, count2, commonality1, commonality2, delta = row
        print(f"{tag:<40} {count1:<8} {count2:<8} {commonality1:<15.6f} {commonality2:<15.6f} {delta:<+15.6f}")
    """
    # Return both tags and their delta values instead of just tags
    result = []
    for row in comparison:
        tag, count1, count2, commonality1, commonality2, delta = row
        delta = delta * 100
        result.append((tag, delta))
    # print(result)
    return result

def get_tag_scores_from_external(tags):
    """
    Get score data from ExternalTagScores table for a list of tags.

    Args:
        tags (list): List of tag strings

    Returns:
        dict: Dictionary mapping tags to their scores (or None if not found)
    """
    try:
        import sqlite3
        # Get the database path from the settings
        from src.ui.settings_manager import load_settings
        settings = load_settings()
        db_path = settings.get("db_path", "mydb.db")

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create a dictionary to store tag scores
        tag_scores = {}

        # Check if we have any tags to look up
        if not tags:
            return tag_scores

        # Get all unique tags (in case there are duplicates)
        unique_tags = list(set(tags))

        # Prepare the SQL query with placeholders
        placeholders = ','.join(['?'] * len(unique_tags))
        query = f"SELECT tag, score FROM ExternalTagScores WHERE tag IN ({placeholders})"

        # Execute the query
        cursor.execute(query, unique_tags)
        results = cursor.fetchall()

        # Map results to a dictionary
        for row in results:
            tag_scores[row[0]] = row[1]

        return tag_scores

    except Exception as e:
        print(f"Error getting tag scores: {e}")
        return {}
    finally:
        if conn:
            conn.close()

def main():
    # Example usage (edit the queries / client to match your setup):
    # RecommendTags(["example tag"], tag_service_file_search="my tags", limit=40, client_type="client1")

    # Define the queries to compare
    query1 = ["system:archive", "system:limit is 3000"]
    query2 = ["l:*", "system:limit is 3000"]
    tag_service_file_search = "my tags"
    tag_service_file_tags = "my tags"

    # Define blacklist and optional whitelist patterns
    blacklist = ["set:*", "comic:*", "marker:*", "page:*", "page#:*"]
    whitelist = None  # Set to None by default, or uncomment below to use
    # whitelist = ["character:*"]  # Only include character tags

    # Compare the queries
    print("\nComparing tag distributions between queries...")
    comparison = compare_queries(query1, query2, tag_service_file_tags, blacklist, whitelist, limit=20)

    if not comparison:
        print("Failed to compare queries")
        return

    """
    # Display the comparison table (commented out for performance)
    print("\nTop Tags by Commonality Delta:")
    print(f"{'Tag':<40} {'Count1':<8} {'Count2':<8} {'Commonality1':<15} {'Commonality2':<15} {'Delta':<15}")
    print("-" * 95)

    for row in comparison[:3]:  # Only show first 3 tags
        tag, count1, count2, commonality1, commonality2, delta = row
        print(f"{tag:<40} {count1:<8} {count2:<8} {commonality1:<15.6f} {commonality2:<15.6f} {delta:<+15.6f}")

    print(comparison)
    """

def parse_query_tags_from_string(query):
    """
    Parse query text to extract tags, supporting both AND and OR syntax.

    Supports:
    - AND query: "tag1, tag2, tag3" (comma-separated)
    - OR query: "[tag1, tag2, tag3]" or '["tag1", "tag2", "tag3"]' (bracket notation)

    Args:
        query (list or str): The query - can be a list of tags or a string

    Returns:
        list: List of tags with proper OR/AND formatting for Hydrus API
    """
    if isinstance(query, list):
        return query

    if not query or not query.strip():
        return []

    query_text = query.strip()

    # Check for OR query syntax: [tag1, tag2, tag3] or ["tag1", "tag2", "tag3"]
    import re
    or_match = re.match(r'^\s*\[(.*)\]\s*$', query_text)
    if or_match:
        # Extract content inside brackets
        content = or_match.group(1).strip()

        # Check if it's quoted JSON style: ["tag1", "tag2"]
        if content.startswith('"') and content.endswith('"'):
            # Parse JSON-style quoted tags
            try:
                import json
                tags = json.loads(f'[{content}]')
                return tags
            except:
                pass

        # Parse comma-separated tags inside brackets
        tags = [tag.strip().strip('"') for tag in content.split(',')]
        return tags

    # Default AND query: comma-separated tags
    tags = [tag.strip() for tag in query_text.split(',')]
    return tags

if __name__ == "__main__":
    main()