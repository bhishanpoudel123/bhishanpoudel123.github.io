import json
import os
from pathlib import Path

def generate_quiz_html(json_file_path, output_html_path):
    """Generate an HTML quiz file from a JSON question file."""
    
    # Load the JSON data
    with open(json_file_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    # Create the HTML content
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{Path(json_file_path).parent.name} Quiz</title>
    <link rel="stylesheet" href="../styles.css">
</head>
<body>
    <div class="container">
        <h1>{Path(json_file_path).parent.name} Quiz</h1>
        <div class="quiz-container">
    """
    
    for i, question in enumerate(questions, 1):
        options_html = "\n".join(
            f'<div class="option"><input type="radio" name="q{i}" id="q{i}o{oi}" value="{oi}">'
            f'<label for="q{i}o{oi}">{option}</label></div>'
            for oi, option in enumerate(question['options'])
        )
        
        # Check if there are long answer markdown files
        long_answer_ref = ""
        if 'answer_long_md' in question and question['answer_long_md']:
            long_answer_path = question['answer_long_md'][0].replace("data/", "../data/")
            long_answer_ref = f'<p class="long-answer-ref"><a href="{long_answer_path}" target="_blank">Detailed Explanation</a></p>'
        
        html_content += f"""
            <div class="question" id="q{i}">
                <h3>Question {i}: {question['question']}</h3>
                <div class="options">
                    {options_html}
                </div>
                <div class="answer hidden">
                    <p><strong>Answer:</strong> {question['answer']}</p>
                    <p><strong>Explanation:</strong> {question['explanation']}</p>
                    {long_answer_ref}
                </div>
                <button class="show-answer">Show Answer</button>
            </div>
        """
    
    html_content += """
        </div>
    </div>
    <script src="../quiz.js"></script>
</body>
</html>
    """
    
    # Write the HTML file
    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

def process_all_quiz_files(data_dir='data'):
    """Process all JSON quiz files in the data directory."""
    data_path = Path(data_dir)
    
    for topic_dir in data_path.iterdir():
        if topic_dir.is_dir():
            json_file = topic_dir / f"{topic_dir.name.lower()}_questions.json"
            if json_file.exists():
                html_file = topic_dir / f"{topic_dir.name.lower()}.html"
                print(f"Processing {json_file} -> {html_file}")
                generate_quiz_html(json_file, html_file)

if __name__ == "__main__":
    process_all_quiz_files()