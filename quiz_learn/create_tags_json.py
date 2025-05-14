#!/usr/bin/env python3
import os
import json
from pathlib import Path
from collections import defaultdict


def get_question_files(data_dir="data"):
    """Find all question JSON files in the data directory"""
    question_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.startswith("qn_") and file.endswith(".json"):
                question_files.append(Path(root) / file)
    return question_files


def extract_tags_from_file(file_path):
    """Extract tags from a question JSON file"""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            # Handle both array and single question formats
            questions = data if isinstance(data, list) else [data]
            tags = set()
            for q in questions:
                if "tags" in q and isinstance(q["tags"], list):
                    tags.update(tag.strip() for tag in q["tags"])
            return sorted(tags)
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Error processing {file_path}: {str(e)}")
        return []


def create_tags_json(data_dir="data", output_file="data/tags.json"):
    """Create a JSON file with all unique tags from all questions"""
    question_files = get_question_files(data_dir)
    if not question_files:
        print(f"Error: No question files found in {data_dir}")
        return False

    # Collect all unique tags and count their occurrences
    tag_counts = defaultdict(int)
    tag_sources = defaultdict(list)

    for q_file in question_files:
        tags = extract_tags_from_file(q_file)
        for tag in tags:
            tag_counts[tag] += 1
            tag_sources[tag].append(q_file.name)

    # Sort tags by count (descending) then alphabetically
    sorted_tags = sorted(tag_counts.keys(), key=lambda x: (-tag_counts[x], x.lower()))

    # Prepare the output
    output = {
        "tags": sorted_tags,
        "tag_details": {
            tag: {
                "count": tag_counts[tag],
                "source_files": tag_sources[tag][:5],  # Show first 5 files only
            }
            for tag in sorted_tags
        },
        "total_questions": sum(tag_counts.values()),
        "unique_tags": len(sorted_tags),
        "generated_by": "create_tags_json.py",
    }

    # Write to file
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)

    print(f"Successfully created {output_path}")
    print(
        f"Found {len(sorted_tags)} unique tags across {len(question_files)} question files"
    )
    return True


if __name__ == "__main__":
    create_tags_json()
