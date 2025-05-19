#!/usr/bin/env python3
"""
Script to create index.json files for all data folders in the quiz project.
This script scans the data directory and creates index.json files for each subfolder
containing the list of JSON files present in that folder.
"""

import os
import json
from pathlib import Path


def create_index_json():
    """
    Create index.json files for all subdirectories in the data folder.
    Each index.json will contain a list of JSON files in that directory.
    """
    # Get the current directory (should be the quiz root folder)
    root_dir = Path(__file__).parent
    data_dir = root_dir / "data"

    if not data_dir.exists():
        print(f"Error: 'data' directory not found at {data_dir}")
        return

    # Iterate through all subdirectories in the data folder
    for folder_path in data_dir.iterdir():
        if folder_path.is_dir():
            # Skip special directories
            if folder_path.name.startswith("."):
                continue

            # Find all JSON files in the current folder
            json_files = []
            for file_path in folder_path.iterdir():
                if file_path.is_file() and file_path.suffix.lower() == ".json":
                    # Don't include index.json itself
                    if file_path.name != "index.json":
                        json_files.append(file_path.name)

            # Sort the files for consistent ordering
            json_files.sort()

            # Create the index.json content
            index_content = {"files": json_files}

            # Write the index.json file
            index_file_path = folder_path / "index.json"
            try:
                with open(index_file_path, "w", encoding="utf-8") as f:
                    json.dump(index_content, f, indent=4)

                print(f"Created {index_file_path} with {len(json_files)} files")
                if json_files:
                    print(f"  Files: {', '.join(json_files)}")
                else:
                    print("  Warning: No JSON files found in this directory")
            except Exception as e:
                print(f"Error creating {index_file_path}: {e}")


def validate_json_files():
    """
    Validate that all JSON files are properly formatted.
    """
    root_dir = Path(__file__).parent
    data_dir = root_dir / "data"

    if not data_dir.exists():
        print(f"Error: 'data' directory not found at {data_dir}")
        return

    print("\nValidating JSON files...")
    errors = []

    for folder_path in data_dir.iterdir():
        if folder_path.is_dir():
            for file_path in folder_path.iterdir():
                if file_path.is_file() and file_path.suffix.lower() == ".json":
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            json.load(f)
                    except json.JSONDecodeError as e:
                        errors.append(f"{file_path}: {e}")
                    except Exception as e:
                        errors.append(f"{file_path}: Error reading file - {e}")

    if errors:
        print("JSON validation errors found:")
        for error in errors:
            print(f"  {error}")
    else:
        print("All JSON files are valid!")


def main():
    """
    Main function to run the script.
    """
    print("Creating index.json files for quiz data directories...")
    print("=" * 50)

    create_index_json()

    print("\n" + "=" * 50)
    validate_json_files()

    print("\nDone! Index files created.")
    print("\nNote: Make sure your JSON files have proper structure with:")


if __name__ == "__main__":
    main()
