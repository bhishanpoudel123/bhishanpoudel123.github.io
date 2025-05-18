import json
import datetime
from pathlib import Path
import textwrap
import sys

def create_topic_readmes(data_dir="data"):
    """Generate README.md files for each topic directory containing question JSON files."""
    data_path = Path(data_dir)
    print("Generating topic README files...\n")

    if not data_path.exists():
        print(f"❌ Error: Data directory '{data_dir}' not found")
        return False

    success_count = 0
    error_count = 0

    for topic_dir in sorted(data_path.iterdir()):
        if not topic_dir.is_dir() or topic_dir.name.startswith("."):
            continue

        json_files = list(topic_dir.glob("*_questions.json"))
        if not json_files:
            print(f"⚠️  No question file found in: {topic_dir.name}")
            error_count += 1
            continue

        json_path = json_files[0]
        readme_path = topic_dir / "README.md"

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                questions = json.load(f)

            if not isinstance(questions, list):
                raise ValueError("Questions data should be a list")

            md_content = generate_markdown(topic_dir.name, questions, json_path.name)

            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(md_content)

            print(f"✅ Created README for: {topic_dir.name} ({len(questions)} questions)")
            success_count += 1

        except json.JSONDecodeError as e:
            print(f"❌ JSON error in {topic_dir.name}: {str(e)}")
            error_count += 1
        except Exception as e:
            print(f"❌ Error processing {topic_dir.name}: {str(e)}")
            error_count += 1

    print(f"\n✨ Completed: {success_count} successful, {error_count} errors")
    return error_count == 0

def generate_markdown(topic_name, questions, json_filename):
    """Generate markdown content for a topic's README file."""
    clean_topic = topic_name.replace("_", " ").title()
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    
    # Generate table of contents
    toc_entries = []
    question_sections = []
    
    for idx, q in enumerate(questions, 1):
        qn_number = f"{idx:02d}"
        anchor = f"q{qn_number}"
        question_text = q.get("question", "").strip()
        
        # Validate required fields
        if not question_text:
            raise ValueError(f"Question {idx} is missing text")
        if "answer" not in q:
            raise ValueError(f"Question {idx} is missing answer")
        
        # TOC entry
        toc_entries.append(f"- [Qn {qn_number}: {question_text}](#{anchor})")
        
        # Build question section
        section = [
            f'### <a id="{anchor}"></a> Qn {qn_number}',
            "",
            f"**Question**  \n{question_text}",
            ""
        ]
        
        # Add options if they exist
        if "options" in q and q["options"]:
            section.append("**Options**  \n")
            for i, opt in enumerate(q["options"], 1):
                section.append(f"{i}. {opt}  ")
            section.append("")
        
        # Add answer and explanation
        section.extend([
            f"**Answer**  \n{q['answer']}",
            ""
        ])
        
        if "explanation" in q and q["explanation"]:
            explanation = textwrap.fill(
                q["explanation"], 
                width=80, 
                subsequent_indent="  ",
                replace_whitespace=False
            )
            section.extend([
                "**Explanation**  ",
                f"{explanation}",
                ""
            ])
        
        # Add link to detailed explanation if available
        if "answer_long_md" in q and q["answer_long_md"]:
            md_path = Path(q["answer_long_md"][0])
            section.extend([
                "**Detailed Explanation**  ",
                f"See detailed documentation: [{md_path.name}]({md_path})",
                ""
            ])
        
        # Add navigation link
        section.append("[↑ Go to TOC](#toc)\n")
        question_sections.append("\n".join(section))
    
    # Build the final markdown content without problematic f-string
    markdown_content = [
        f"# {clean_topic} Study Guide <a id=\"toc\"></a>",
        "",
        "## Table of Contents",
        "  \n".join(toc_entries),
        "",
        "## Questions",
        "\n\n".join(question_sections),
        "",
        "---",
        "",
        f"*Automatically generated from [{json_filename}]({json_filename})*",
        f"*Updated: {now}*",
        ""
    ]
    
    return "\n".join(markdown_content)

if __name__ == "__main__":
    try:
        success = create_topic_readmes()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        sys.exit(1)