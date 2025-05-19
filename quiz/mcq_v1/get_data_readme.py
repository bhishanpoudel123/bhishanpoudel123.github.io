import os
import json

# Define the path to the data folder
DATA_FOLDER = "data"
README_FILE = os.path.join(DATA_FOLDER, "README.md")


def get_topics_summary():
    """Scan all topic folders in 'data' and summarize the number of questions in each JSON file."""
    topics_summary = {}

    # Iterate over directories in the data folder
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
                        questions  # Store all questions
                    )

    return topics_summary


def generate_readme(topics_summary):
    """Generate README.md with Table of Contents, Summary, and detailed questions."""
    content = []

    # Table of Contents
    content.append("# Table of Contents\n")
    for topic in topics_summary:
        anchor = topic.lower().replace(" ", "-")
        content.append(f"- [{topic}](#{anchor})")

    # Summary section
    content.append("\n## Summary\n")
    for topic, questions in topics_summary.items():
        content.append(f"- **{topic}**: {len(questions)} questions")

    # Detailed question sections with numbered questions
    for topic_folder, questions in topics_summary.items():
        content.append(f"\n# {topic_folder}\n")

        for i, question in enumerate(questions, start=1):
            content.append(
                f"### Qn {i:02d}: {question['question']}\n"
            )  # Adds question number formatted as 'Qn 01'
            content.append("<p><details>")
            content.append(f"<summary>Click to reveal answer</summary>\n")
            content.append(f"{question['answer']}\n")
            if "explanation" in question and question["explanation"]:
                content.append(f"\n**Explanation:** {question['explanation']}\n")
            content.append("</details></p>\n")

    # Write to README.md
    with open(README_FILE, "w", encoding="utf-8") as file:
        file.write("\n".join(content))


if __name__ == "__main__":
    topics_summary = get_topics_summary()
    generate_readme(topics_summary)
    print(f"README.md generated successfully in '{DATA_FOLDER}'")
