import os
import json
from pathlib import Path


def create_html_index():
    data_dir = Path("data")
    html_index = {}

    print("Building HTML index...\n")

    for topic_dir in data_dir.iterdir():
        if not topic_dir.is_dir() or topic_dir.name.startswith("."):
            continue

        # Get human-readable category name
        category = topic_dir.name.replace("_", " ")
        html_files = []

        # Collect all HTML files regardless of name
        for file in topic_dir.glob("*.html"):
            if file.is_file():
                html_files.append(file.name)

        if html_files:
            html_index[category] = html_files
            print(f"✅ Added: {category.ljust(25)} → {html_files}")
        else:
            print(f"⚠️  No HTML files found in: {topic_dir.name}")

    # Save to index_html.json
    index_path = data_dir / "index_html.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(html_index, f, indent=2, ensure_ascii=False)

    print(f"\nIndex created with {len(html_index)} entries at: {index_path}")


if __name__ == "__main__":
    create_html_index()
