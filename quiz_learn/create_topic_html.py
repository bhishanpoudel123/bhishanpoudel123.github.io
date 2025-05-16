import os
import json

# Define base path
DATA_FOLDER = "data"


def get_topics_summary():
    """Scan all topic folders in 'data' and retrieve all questions."""
    topics_summary = {}

    for topic_folder in sorted(os.listdir(DATA_FOLDER)):
        topic_path = os.path.join(DATA_FOLDER, topic_folder)

        if os.path.isdir(topic_path):
            json_files = [f for f in os.listdir(topic_path) if f.endswith(".json")]

            if json_files:
                json_file_path = os.path.join(
                    topic_path, json_files[0]
                )  # Assuming one JSON per folder
                with open(json_file_path, "r", encoding="utf-8") as file:
                    questions = json.load(file)
                    topics_summary[topic_folder.replace("_", " ")] = (
                        topic_path,
                        questions,
                    )  # Store folder path & questions

    return topics_summary


def generate_html_files(topics_summary):
    """Generate individual HTML files for each topic inside its own folder."""
    for topic_folder, (folder_path, questions) in topics_summary.items():
        # Correcting file path: Ensure HTML file is created **inside the topic folder**
        file_name = f"{topic_folder.lower().replace(' ', '_')}.html"
        file_path = os.path.join(folder_path, file_name)

        content = [
            f"<html>\n<head><title>{topic_folder}</title></head>\n<body>",
            f"<h1>{topic_folder}</h1>",
        ]

        for i, question in enumerate(questions, start=1):
            content.append(f"<h2>Qn {i:02d}: {question['question']}</h2>")
            content.append(
                '<div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">'
            )
            content.append(f"<strong>Answer:</strong> {question['answer']}")
            if "explanation" in question and question["explanation"]:
                content.append(
                    f"<br><strong>Explanation:</strong> {question['explanation']}"
                )
            content.append("</div>\n")  # Close the styled div

        content.append("</body>\n</html>")

        # Write to HTML file
        with open(file_path, "w", encoding="utf-8") as file:
            file.write("\n".join(content))

    print("HTML files generated successfully in their respective topic folders!")


if __name__ == "__main__":
    topics_summary = get_topics_summary()
    generate_html_files(topics_summary)
