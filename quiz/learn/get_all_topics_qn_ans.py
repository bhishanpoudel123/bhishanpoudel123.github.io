import os
import json

# Define paths
DATA_FOLDER = "data"
QUESTIONS_FILE = os.path.join(DATA_FOLDER, "all_topics_questions_answers.md")


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
                        questions  # Store all questions
                    )

    return topics_summary


def generate_questions_file(topics_summary):
    """Generate all_topics_questions_answers.md with Table of Contents and detailed questions/answers."""
    content = ["# All Topics: Questions & Answers\n"]

    # **Table of Contents**
    content.append("\n## Table of Contents\n")
    for topic in topics_summary.keys():
        anchor = topic.lower().replace(" ", "-")
        content.append(f"- [{topic}](#{anchor})")

    # **Detailed Questions & Answers**
    for topic_folder, questions in topics_summary.items():
        anchor = topic_folder.lower().replace(" ", "-")
        content.append(f"\n## {topic_folder}\n")

        for i, question in enumerate(questions, start=1):
            content.append(f"### Qn {i:02d}: {question['question']}\n")
            content.append(f"**Answer:** {question['answer']}\n")
            if "explanation" in question and question["explanation"]:
                content.append(f"**Explanation:** {question['explanation']}\n")

    # Write to markdown file
    with open(QUESTIONS_FILE, "w", encoding="utf-8") as file:
        file.write("\n".join(content))


if __name__ == "__main__":
    topics_summary = get_topics_summary()
    generate_questions_file(topics_summary)
    print(
        f"'all_topics_questions_answers.md' generated successfully in '{DATA_FOLDER}'"
    )
