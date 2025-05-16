import json
import datetime
from pathlib import Path
import textwrap


def create_topic_readmes():
    data_dir = Path("data")
    print("Generating topic README files...\n")

    for topic_dir in data_dir.iterdir():
        if not topic_dir.is_dir() or topic_dir.name.startswith("."):
            continue

        json_files = list(topic_dir.glob("*_questions.json"))
        if not json_files:
            print(f"⚠️  No question file found in: {topic_dir.name}")
            continue

        json_path = json_files[0]
        readme_path = topic_dir / "README.md"

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                questions = json.load(f)

            md_content = generate_markdown(topic_dir.name, questions, json_path.name)

            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(md_content)

            print(
                f"✅ Created README for: {topic_dir.name} ({len(questions)} questions)"
            )

        except Exception as e:
            print(f"❌ Error processing {topic_dir.name}: {str(e)}")


def generate_markdown(topic_name, questions, json_filename):
    clean_topic = topic_name.replace("_", " ").title()
    toc_entries = []
    question_sections = []

    for idx, q in enumerate(questions, 1):
        qn_number = f"{idx:02d}"
        anchor = f"q{qn_number}"
        question_text = q.get("question", "").strip()

        # TOC entry with full question
        toc_entries.append(f"- [Qn {qn_number}: {question_text}](#{anchor})")

        # Build question section
        section = f'### <a id="{anchor}"></a> Qn {qn_number}\n\n'
        section += f"**Question**  \n{question_text}\n\n"

        if "options" in q:
            section += "**Options**  \n"
            for i, opt in enumerate(q["options"], 1):
                section += f"{i}. {opt}  \n"
            section += "\n"

        section += f"**Answer**  \n{q.get('answer', '')}\n\n"

        if "explanation" in q:
            section += "**Explanation**  \n"
            explanation = textwrap.fill(
                q["explanation"], width=80, subsequent_indent="  "
            )
            section += f"{explanation}\n\n"

        if "answer_long_md" in q:
            section += "**Detailed Explanation**  \n"
            section += f"See detailed documentation: [{Path(q['answer_long_md'][0]).name}]({q['answer_long_md'][0]})\n\n"

        # Add TOC navigation link
        section += "[↑ Go to TOC](#toc)\n\n"
        question_sections.append(section)

    return f"""# {clean_topic} Study Guide <a id="toc"></a>

## Table of Contents
{"  \n".join(toc_entries)}

## Questions
{"  \n\n".join(question_sections)}

---

*Automatically generated from [{json_filename}]({json_filename})*  
*Updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}*
"""


if __name__ == "__main__":
    create_topic_readmes()
    print("\n✨ All README files generated!")
