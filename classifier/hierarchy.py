"""
Utilities for working with topic hierarchies.

This module provides functions for loading, navigating, and manipulating
hierarchical topic structures.
"""

import json
from pathlib import Path
from typing import Dict, List, Any


def load_topic_hierarchy(filepath: str | Path) -> Dict[str, Any] | None:
    """
    Load a topic hierarchy from a JSON file.
    
    Args:
        filepath: Path to the JSON file containing the hierarchy
        
    Returns:
        Dictionary representing the hierarchy, or None if loading fails
    """
    try:
        with open(str(filepath), "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: The file {filepath} is not a valid JSON file.")
        return None


def get_node_path(tree: Dict[str, Any], path: List[str]) -> List[Dict[str, Any]] | None:
    """
    Extract the sequence of nodes along a given path in the hierarchy.
    
    This function traverses the tree following the provided path and returns
    the configuration for each node along the way.
    
    Args:
        tree: Root node of the hierarchy
        path: List of node names forming the path (e.g., ["Root", "A", "B"])
        
    Returns:
        List of node configurations along the path, or None if path doesn't exist
        
    Example:
        >>> tree = {"name": "Root", "children": [{"name": "A", "children": []}]}
        >>> get_node_path(tree, ["Root", "A"])
        [{"name": "A", ...}]
    """
    node = tree
    nodes_along_path = []
    
    # Skip root if it's the first element in path
    start_idx = 0
    if path and path[0] == node.get("name"):
        start_idx = 1

    for element in path[start_idx:]:
        children = node.get("children", [])
        child_node = next((c for c in children if c["name"] == element), None)
        
        if not child_node:
            return None
            
        nodes_along_path.append({
            "name": child_node.get("name"),
            "definition": child_node.get("definition"),
            "description": child_node.get("description"),
            "keywords": child_node.get("keywords", []),
            "scope": child_node.get("scope", "[None]")
        })
        node = child_node

    return nodes_along_path


def get_all_leaf_paths(tree: Dict[str, Any], separator: str = ">") -> List[str]:
    """
    Get all paths from root to leaf nodes.
    
    Args:
        tree: Root node of the hierarchy
        separator: String to use for joining path components
        
    Returns:
        List of path strings (e.g., ["Root>A>B", "Root>C>D"])
    """
    paths = []
    root_name = tree.get("name", "[ROOT]")
    
    def _traverse(node: Dict[str, Any], current_path: str):
        children = node.get("children", [])
        if not children:
            # Leaf node
            paths.append(current_path)
        else:
            for child in children:
                child_name = child.get("name", "")
                new_path = f"{current_path}{separator}{child_name}"
                _traverse(child, new_path)
    
    _traverse(tree, root_name)
    return paths


def build_tree_from_paths(paths: List[str], separator: str = ">") -> Dict[str, Any]:
    """
    Build a nested tree structure from a list of path strings.
    
    Useful for visualizing classification results.
    
    Args:
        paths: List of path strings (e.g., ["Root>A>B", "Root>C"])
        separator: String used to separate path components
        
    Returns:
        Nested dictionary representing the tree structure
        
    Example:
        >>> paths = ["Root>A>B", "Root>A>C", "Root>D"]
        >>> tree = build_tree_from_paths(paths)
        >>> tree
        {"Root": {"children": {"A": {"children": {"B": ..., "C": ...}}, "D": ...}}}
    """
    tree = {}
    for path in paths:
        parts = path.split(separator)
        current_level = tree
        for part in parts:
            if part not in current_level:
                current_level[part] = {"children": {}}
            current_level = current_level[part]["children"]
    return tree


def format_tree_as_string(tree: Dict[str, Any], prefix: str = "") -> str:
    """
    Format a tree structure as an ASCII tree diagram.
    
    Args:
        tree: Nested dictionary from build_tree_from_paths
        prefix: Prefix for indentation (used internally for recursion)
        
    Returns:
        String representation of the tree with ASCII art
        
    Example:
        ├── A
        │   ├── B
        │   └── C
        └── D
    """
    lines = []
    
    def _format_level(sub_tree: Dict[str, Any], pfx: str = ""):
        children = sorted(sub_tree.items())
        for i, (name, data) in enumerate(children):
            is_last = i == len(children) - 1
            connector = "└── " if is_last else "├── "
            lines.append(f"{pfx}{connector}{name}")
            _format_level(
                data["children"], 
                pfx + ("    " if is_last else "│   ")
            )
    
    _format_level(tree, prefix)
    return "\n".join(lines)


def validate_hierarchy(tree: Dict[str, Any]) -> List[str]:
    """
    Validate a hierarchy structure and return any issues found.
    
    Args:
        tree: Root node of the hierarchy
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    def _validate_node(node: Dict[str, Any], path: str):
        # Check required fields
        if "name" not in node:
            errors.append(f"Missing 'name' field at path: {path}")
        
        # Check for duplicate child names
        children = node.get("children", [])
        child_names = [c.get("name") for c in children]
        duplicates = [n for n in child_names if child_names.count(n) > 1]
        if duplicates:
            errors.append(
                f"Duplicate child names at {path}: {set(duplicates)}"
            )
        
        # Recurse to children
        for child in children:
            child_path = f"{path}>{child.get('name', '?')}"
            _validate_node(child, child_path)
    
    _validate_node(tree, tree.get("name", "[ROOT]"))
    return errors
