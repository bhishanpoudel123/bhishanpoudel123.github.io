import markdown
from pathlib import Path

# File paths
md_file = Path("README.md")
html_file = Path("README.html")

# Read the markdown file
text = md_file.read_text(encoding="utf-8")

# Convert markdown to HTML
html_body = markdown.markdown(text, extensions=["fenced_code", "codehilite", "tables"])

# Full HTML with responsive design
html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>README</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 1rem;
            line-height: 1.6;
            background: #fff;
            color: #333;
            max-width: 800px;
            margin: auto;
        }}
        pre code {{
            background-color: #f4f4f4;
            padding: 1em;
            display: block;
            overflow-x: auto;
        }}
        code {{
            background-color: #f4f4f4;
            padding: 0.2em 0.4em;
            border-radius: 4px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
        }}
        th, td {{
            padding: 0.5em;
            border: 1px solid #ccc;
            text-align: left;
        }}
        @media (max-width: 600px) {{
            body {{
                padding: 0.5rem;
            }}
        }}
    </style>
</head>
<body>
{html_body}
</body>
</html>
"""

# Write the HTML to file
html_file.write_text(html_template, encoding="utf-8")
print(f"Converted '{md_file}' to '{html_file}'")
