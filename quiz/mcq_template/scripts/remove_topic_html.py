import os

# Define the main folder
main_folder = "data"

# Iterate through each subfolder inside the main folder
for subfolder in os.listdir(main_folder):
    subfolder_path = os.path.join(main_folder, subfolder)

    # Ensure it's a directory
    if os.path.isdir(subfolder_path):
        # Convert folder name to lowercase
        lower_name = subfolder.lower()

        # Construct the expected HTML file path
        html_file = os.path.join(subfolder_path, f"{lower_name}.html")

        # Check if the HTML file exists and remove it
        if os.path.isfile(html_file):
            print(f"Removing: {html_file}")
            os.remove(html_file)
