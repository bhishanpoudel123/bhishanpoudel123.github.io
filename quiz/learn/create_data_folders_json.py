#!/usr/bin/env python3
import os
import json
from pathlib import Path


def is_valid_question_folder(folder_path):
    """Check if a folder contains at least one question JSON file"""
    for file in os.listdir(folder_path):
        if file.startswith("qn_") and file.endswith(".json"):
            return True
    return False


def create_data_folders_json(data_dir="data", output_file="data/data_folders.json"):
    """Create a JSON file listing all valid question folders"""
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Error: Data directory '{data_dir}' not found")
        return False

    # Get all subdirectories in data folder
    folders = [d.name for d in data_path.iterdir() if d.is_dir()]

    # Filter to only include folders with question files
    question_folders = [
        folder for folder in folders if is_valid_question_folder(data_path / folder)
    ]

    # Create the output
    output = {
        "folders": sorted(question_folders),
        "generated_by": "create_data_folder_json.py",
    }

    # Write to file
    output_path = data_path / "data_folders.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Successfully created {output_path} with {len(question_folders)} folders")
    return True


if __name__ == "__main__":
    create_data_folders_json()
