import json
import os

data_directory = 'data'

def add_keys_to_json(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{file_path}'")
        return

    updated_data = []
    for item in data:
        item['answer_long_md'] = []
        item['answer_long_html'] = []
        updated_data.append(item)

    with open(file_path, 'w') as f:
        json.dump(updated_data, f, indent=4)

    print(f"Successfully added 'answer_long_md' and 'answer_long_html' keys to all items in '{file_path}'.")

# Ensure the 'data' directory exists
if not os.path.exists(data_directory):
    print(f"Warning: Directory '{data_directory}' not found.")
else:
    # Iterate through all items in the 'data' directory
    for item_name in os.listdir(data_directory):
        item_path = os.path.join(data_directory, item_name)

        # Check if the item is a directory
        if os.path.isdir(item_path):
            print(f"Looking into folder: '{item_path}'")
            # Iterate through all files in the subdirectory
            for filename in os.listdir(item_path):
                if filename.endswith('.json'):
                    file_path = os.path.join(item_path, filename)
                    add_keys_to_json(file_path)
        # Check if the item in 'data' is a json file directly
        elif item_name.endswith('.json'):
            file_path = os.path.join(data_directory, item_name)
            add_keys_to_json(file_path)

print("Finished processing JSON files.")