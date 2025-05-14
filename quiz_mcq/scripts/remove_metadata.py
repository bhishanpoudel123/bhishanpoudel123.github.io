import json
import os


def clean_json_files(data_folder):
    # Get all JSON files in the data folder except index.json
    json_files = [
        f for f in os.listdir(data_folder) if f.endswith(".json") and f != "index.json"
    ]

    for json_file in json_files:
        file_path = os.path.join(data_folder, json_file)

        # Read the JSON file
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error reading {json_file}: {e}")
                continue

        # Process each question to remove metadata if it exists
        processed_data = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    # Create new item without metadata
                    new_item = {k: v for k, v in item.items() if k != "metadata"}
                    processed_data.append(new_item)
                else:
                    processed_data.append(item)
        else:
            print(f"{json_file} doesn't contain a list of questions, skipping")
            continue

        # Write the processed data back to the same file
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)

        print(f"Processed {json_file} - removed metadata")


if __name__ == "__main__":
    data_folder = "data"  # Folder containing JSON files
    if os.path.exists(data_folder):
        clean_json_files(data_folder)
        print("Processing complete!")
    else:
        print(f"Folder '{data_folder}' not found")
