import os
import json

data_dir = 'data'
index_data = {
    "files": {},
    "generated_by": "create_index_json.py"
}

# Traverse all subdirectories in 'data'
for folder_name in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, folder_name)

    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                json_path = os.path.join(folder_path, filename)
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list) and data:
                            category = data[0].get("category", folder_name.replace('_', ' '))
                            index_data["files"][category] = filename
                            print(f"Added: {category} → {filename}")
                except Exception as e:
                    print(f"Error reading {json_path}: {e}")

# Write index.json to the data directory
index_file_path = os.path.join(data_dir, 'index.json')
with open(index_file_path, 'w', encoding='utf-8') as f:
    json.dump(index_data, f, indent=2, ensure_ascii=False)

print(f"\n✅ Created index at: {index_file_path}")
