import os
import json

data_dir = "data"
exclude_files = ["index.json"]

# Get all JSON files in data directory
files = [
    f for f in os.listdir(data_dir) if f.endswith(".json") and f not in exclude_files
]

# Create index.json content
index_content = {"files": sorted(files), 'generated_by': 'create_index_json.py'}

# Write to file
with open(os.path.join(data_dir, "index.json"), "w") as f:
    json.dump(index_content, f, indent=2)

print("Created index.json with files:", files)
